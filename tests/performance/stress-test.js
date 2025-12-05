// AI Audio Upscaler Pro - Stress Testing with k6
// High-load stress testing to determine system limits

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const successfulRequests = new Counter('successful_requests');
const failedRequests = new Counter('failed_requests');

// Stress test configuration
export const options = {
  stages: [
    // Stress test stages
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '3m', target: 200 },  // Ramp up to 200 users (stress level)
    { duration: '2m', target: 300 },  // Push to 300 users (breaking point)
    { duration: '1m', target: 400 },  // Push to 400 users (beyond capacity)
    { duration: '5m', target: 400 },  // Hold at 400 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(50)<1000', 'p(95)<5000'], // Relaxed thresholds for stress test
    http_req_failed: ['rate<0.5'],                    // Allow up to 50% failures under stress
    errors: ['rate<0.6'],                             // Allow up to 60% error rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Simple high-frequency requests to stress the system

  // Test 1: Health endpoint (lightweight)
  const healthResponse = http.get(`${BASE_URL}/health`, {
    timeout: '10s',
  });

  const healthSuccess = check(healthResponse, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 5s': (r) => r.timings.duration < 5000,
  });

  if (healthSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    errorRate.add(1);
  }

  // Brief pause
  sleep(0.1);

  // Test 2: API root endpoint
  const apiResponse = http.get(`${BASE_URL}/`, {
    timeout: '10s',
  });

  const apiSuccess = check(apiResponse, {
    'root status is 200': (r) => r.status === 200,
    'root response has service name': (r) => {
      try {
        return r.json().service !== undefined;
      } catch (e) {
        return false;
      }
    },
  });

  if (apiSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    errorRate.add(1);
  }

  // Brief pause
  sleep(0.1);

  // Test 3: Readiness check (more comprehensive)
  const readyResponse = http.get(`${BASE_URL}/health/ready`, {
    timeout: '10s',
  });

  const readySuccess = check(readyResponse, {
    'ready status is 200': (r) => r.status === 200,
    'ready response time < 3s': (r) => r.timings.duration < 3000,
  });

  if (readySuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    errorRate.add(1);
  }

  // Variable sleep to create irregular load patterns
  sleep(Math.random() * 0.5);

  // Test 4: Metrics endpoint (if available)
  const metricsResponse = http.get(`${BASE_URL}/metrics`, {
    timeout: '15s',
  });

  const metricsSuccess = check(metricsResponse, {
    'metrics accessible': (r) => [200, 404].includes(r.status), // 404 is acceptable if not exposed
  });

  if (metricsSuccess) {
    successfulRequests.add(1);
  } else {
    failedRequests.add(1);
    errorRate.add(1);
  }

  // Brief pause before next iteration
  sleep(0.1);
}

// Setup function
export function setup() {
  console.log('Starting stress test...');
  console.log(`Target URL: ${BASE_URL}`);

  // Validate service is responding before starting stress test
  const response = http.get(`${BASE_URL}/health`);
  if (response.status !== 200) {
    console.error(`Service not ready for stress test: ${response.status}`);
    return null;
  }

  console.log('Service is ready. Beginning stress test in 5 seconds...');
  return { startTime: Date.now() };
}

// Teardown function
export function teardown(data) {
  if (data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Stress test completed in ${duration}s`);
  }

  console.log('Stress test finished. Check metrics for system behavior under load.');
  console.log('Key indicators to review:');
  console.log('- Response time degradation');
  console.log('- Error rate increase');
  console.log('- System resource utilization');
  console.log('- Recovery time after load reduction');
}