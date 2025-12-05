// AI Audio Upscaler Pro - Load Testing with k6
// Comprehensive performance testing suite

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { FormData } from 'https://jslib.k6.io/formdata/0.0.2/index.js';

// Custom metrics
const authFailures = new Counter('auth_failures');
const uploadFailures = new Counter('upload_failures');
const processingFailures = new Counter('processing_failures');
const authSuccessRate = new Rate('auth_success_rate');
const uploadSuccessRate = new Rate('upload_success_rate');
const processingTime = new Trend('processing_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Hold 10 users
    { duration: '2m', target: 20 },   // Ramp up to 20 users
    { duration: '5m', target: 20 },   // Hold 20 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],        // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],            // Error rate under 10%
    auth_success_rate: ['rate>0.95'],         // Auth success rate over 95%
    upload_success_rate: ['rate>0.90'],       // Upload success rate over 90%
  },
};

// Environment configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_URL = `${BASE_URL}/api/v1`;

// Test data
const TEST_USER = {
  email: 'loadtest@example.com',
  password: 'loadtest123'
};

// Dummy audio file data (base64 encoded WAV header)
const DUMMY_AUDIO_DATA = 'UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';

// Helper function to create authentication token
function authenticate() {
  const loginData = {
    username: TEST_USER.email,
    password: TEST_USER.password
  };

  const response = http.post(`${API_URL}/auth/login`, JSON.stringify(loginData), {
    headers: { 'Content-Type': 'application/json' },
  });

  const success = check(response, {
    'authentication successful': (r) => r.status === 200,
    'has access token': (r) => r.json().access_token !== undefined,
  });

  authSuccessRate.add(success);
  if (!success) {
    authFailures.add(1);
    console.error(`Authentication failed: ${response.status} ${response.body}`);
    return null;
  }

  return response.json().access_token;
}

// Helper function to create form data with audio file
function createAudioUploadForm() {
  const fd = new FormData();

  // Convert base64 to binary
  const binaryString = atob(DUMMY_AUDIO_DATA);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  fd.append('file', http.file(bytes, 'test-audio.wav', 'audio/wav'));
  fd.append('target_sample_rate', '48000');
  fd.append('mode', 'baseline');
  fd.append('use_ai', 'false');

  return fd;
}

export default function () {
  // Test 1: Health Check
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);

  // Test 2: Authentication
  const token = authenticate();
  if (!token) {
    sleep(5); // Wait before retrying
    return;
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  sleep(1);

  // Test 3: API Health Check (Authenticated)
  const apiHealthResponse = http.get(`${API_URL}/health`, { headers });
  check(apiHealthResponse, {
    'API health check status is 200': (r) => r.status === 200,
    'API health check has valid response': (r) => r.json().status !== undefined,
  });

  sleep(1);

  // Test 4: User Profile
  const profileResponse = http.get(`${API_URL}/auth/me`, { headers });
  check(profileResponse, {
    'profile fetch status is 200': (r) => r.status === 200,
    'profile has email': (r) => r.json().email !== undefined,
  });

  sleep(1);

  // Test 5: Usage Statistics
  const usageResponse = http.get(`${API_URL}/auth/usage`, { headers });
  check(usageResponse, {
    'usage stats status is 200': (r) => r.status === 200,
    'usage stats has limits': (r) => r.json().monthly_limit_minutes !== undefined,
  });

  sleep(2);

  // Test 6: Audio Upload and Processing (Baseline mode for speed)
  const uploadStartTime = Date.now();

  try {
    const fd = createAudioUploadForm();
    const uploadHeaders = {
      'Authorization': `Bearer ${token}`,
    };

    const uploadResponse = http.post(`${API_URL}/audio/upload`, fd.body(), {
      headers: Object.assign(uploadHeaders, fd.headers),
    });

    const uploadSuccess = check(uploadResponse, {
      'upload status is 200 or 202': (r) => [200, 202].includes(r.status),
      'upload response has job_id': (r) => {
        try {
          return r.json().job_id !== undefined;
        } catch (e) {
          return false;
        }
      },
    });

    uploadSuccessRate.add(uploadSuccess);

    if (uploadSuccess && uploadResponse.json().job_id) {
      const jobId = uploadResponse.json().job_id;

      // Poll job status
      let jobCompleted = false;
      let attempts = 0;
      const maxAttempts = 30; // 30 attempts with 2s sleep = 60s timeout

      while (!jobCompleted && attempts < maxAttempts) {
        sleep(2);
        attempts++;

        const statusResponse = http.get(`${API_URL}/jobs/${jobId}/status`, { headers });

        const statusCheck = check(statusResponse, {
          'status check is 200': (r) => r.status === 200,
        });

        if (statusCheck) {
          const status = statusResponse.json().status;

          if (status === 'COMPLETED') {
            jobCompleted = true;
            const processingDuration = Date.now() - uploadStartTime;
            processingTime.add(processingDuration);

            check(statusResponse, {
              'job completed successfully': (r) => r.json().status === 'COMPLETED',
              'has output file path': (r) => r.json().output_file_path !== undefined,
            });

          } else if (status === 'FAILED') {
            processingFailures.add(1);
            console.error(`Processing job ${jobId} failed: ${statusResponse.body}`);
            break;
          }
          // Continue polling for PENDING/PROCESSING status
        } else {
          break;
        }
      }

      if (!jobCompleted && attempts >= maxAttempts) {
        processingFailures.add(1);
        console.error(`Processing job ${jobId} timed out after ${maxAttempts * 2}s`);
      }

    } else {
      uploadFailures.add(1);
      console.error(`Upload failed: ${uploadResponse.status} ${uploadResponse.body}`);
    }

  } catch (error) {
    uploadFailures.add(1);
    console.error(`Upload error: ${error.message}`);
  }

  sleep(2);

  // Test 7: Job History
  const historyResponse = http.get(`${API_URL}/jobs?limit=10`, { headers });
  check(historyResponse, {
    'job history status is 200': (r) => r.status === 200,
    'job history is array': (r) => Array.isArray(r.json().jobs),
  });

  sleep(1);

  // Random sleep to simulate realistic user behavior
  sleep(Math.random() * 3 + 1);
}

// Setup function (runs once per VU)
export function setup() {
  console.log(`Starting load test against: ${BASE_URL}`);

  // Test base connectivity
  const response = http.get(`${BASE_URL}/health`);

  if (response.status !== 200) {
    console.error(`Service not ready: ${response.status}`);
    return null;
  }

  console.log('Service is ready for testing');
  return { baseUrl: BASE_URL };
}

// Teardown function (runs once after all VUs finish)
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Base URL: ${data?.baseUrl}`);

  // Summary of custom metrics would be automatically displayed by k6
}