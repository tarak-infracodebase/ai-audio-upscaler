# AI Audio Upscaler Pro - Monitoring & Observability Stack

This directory contains the complete monitoring and observability configuration for the AI Audio Upscaler Pro production environment.

## Architecture Overview

The monitoring stack provides comprehensive observability across:
- Application performance and health
- Infrastructure resources (CPU, Memory, Disk, GPU)
- Database performance
- Queue depth and processing metrics
- Security events and authentication
- Error tracking and alerting

## Components

### 1. Metrics Collection (Prometheus)
- **Location**: `prometheus/prometheus.yml`
- **Purpose**: Collects metrics from all service components
- **Endpoints Monitored**:
  - AI Audio Upscaler API (`/metrics`)
  - Celery Workers
  - PostgreSQL Database
  - Redis Cache
  - Kubernetes nodes and containers
  - Azure Monitor integration

### 2. Visualization (Grafana)
- **Location**: `grafana/dashboards/`
- **Purpose**: Real-time dashboards and visualization
- **Key Dashboards**:
  - Service health overview
  - Request rates and response times
  - Processing job metrics
  - System resource utilization
  - Error rates and patterns
  - GPU memory and utilization

### 3. Alerting (Prometheus Alertmanager)
- **Location**: `alerting/alerts.yaml`
- **Purpose**: Proactive incident detection and notification
- **Alert Categories**:
  - **Critical**: Service down, critical errors, resource exhaustion
  - **Warning**: High latency, resource pressure, queue backlogs
  - **Info**: Traffic patterns, rate limiting activity

### 4. Log Aggregation (Fluentd)
- **Location**: `logging/fluentd-config.yaml`
- **Purpose**: Centralized log collection and analysis
- **Features**:
  - Structured JSON log parsing
  - Kubernetes metadata enrichment
  - Error categorization
  - Azure Log Analytics integration

## Key Metrics

### Application Metrics
```
# Request metrics
http_requests_total{method, endpoint, status_code}
http_request_duration_seconds_bucket{method, endpoint}

# Processing metrics
audio_processing_duration_seconds{status, mode}
audio_files_processed_total{status, mode}
active_jobs_total
queue_depth_total

# Error tracking
processing_errors_total{error_type}
```

### System Metrics
```
# Resource utilization
system_cpu_usage_percent
system_memory_usage_percent
system_disk_usage_percent

# GPU metrics
gpu_memory_usage_percent{device_id}
gpu_utilization_percent{device_id}

# Database metrics
database_connections_active
database_query_duration_seconds{query_type}
```

## Alert Severity Levels

### Critical Alerts
- Service completely down
- Critical error rates (>0.5 errors/sec)
- Resource exhaustion (>95% CPU/Memory)
- Database/Redis unavailable
- GPU out of memory

### Warning Alerts
- High error rates (>0.1 errors/sec)
- Elevated response times (>5s P95)
- Queue backlogs (>100 jobs)
- High resource usage (>80%)
- Processing failures

### Info Alerts
- Unusual traffic patterns
- Rate limiting active
- Low processing throughput

## Deployment

### Prerequisites
```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Create Azure Log Analytics secrets
kubectl create secret generic azure-log-analytics \
  --from-literal=workspace-id=<WORKSPACE_ID> \
  --from-literal=shared-key=<SHARED_KEY> \
  -n ai-upscaler
```

### Deploy Monitoring Stack
```bash
# Apply Fluentd for log aggregation
kubectl apply -f logging/fluentd-config.yaml

# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --values prometheus/values.yaml

# Configure Grafana dashboards
kubectl apply -f grafana/dashboards/
```

### Azure Integration
```bash
# Enable Azure Monitor for AKS
az aks enable-addons \
  --resource-group ai-upscaler-rg \
  --name ai-upscaler-aks \
  --addons monitoring \
  --workspace-resource-id /subscriptions/.../log-analytics-workspace
```

## Grafana Dashboard Access

1. **Local Development**:
   ```bash
   kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
   ```
   Access: http://localhost:3000
   Default credentials: admin/admin

2. **Production**: Use Azure-managed Grafana with SSO integration

## Alert Configuration

### Slack Integration
```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: '<SLACK_WEBHOOK_URL>'
    channel: '#ai-upscaler-alerts'
    title: 'AI Audio Upscaler Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### PagerDuty Integration
```yaml
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: '<PAGERDUTY_SERVICE_KEY>'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

## Log Queries

### Common Kusto Queries for Azure Log Analytics

```kusto
// Error analysis
AIAudioUpscalerLogs
| where TimeGenerated > ago(1h)
| where level_s == "ERROR"
| summarize count() by error_category_s
| order by count_ desc

// Processing performance
AIAudioUpscalerLogs
| where TimeGenerated > ago(1h)
| where processing_duration_d > 0
| summarize
    avg_duration = avg(processing_duration_d),
    p95_duration = percentile(processing_duration_d, 95),
    count = count()
by bin(TimeGenerated, 5m)

// Authentication failures
AIAudioUpscalerLogs
| where TimeGenerated > ago(1h)
| where message_s contains "authentication failed"
| summarize count() by client_ip_s
| order by count_ desc
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
   - Check GPU memory leaks in processing workers
   - Review batch size configuration
   - Monitor cache cleanup processes

2. **Queue Backlogs**:
   - Scale Celery workers horizontally
   - Check GPU node availability
   - Review processing timeouts

3. **Authentication Errors**:
   - Verify Azure B2C configuration
   - Check JWT token expiration
   - Review Key Vault connectivity

### Debug Commands
```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus-server 9090:9090 -n monitoring
# Visit: http://localhost:9090/targets

# View Grafana logs
kubectl logs -l app.kubernetes.io/name=grafana -n monitoring

# Check Fluentd status
kubectl logs -l name=fluentd -n ai-upscaler
```

## Performance Tuning

### Prometheus Optimization
- Adjust scrape intervals based on metric importance
- Use recording rules for expensive queries
- Configure appropriate retention policies

### Log Volume Management
- Filter noisy logs at collection time
- Use sampling for high-frequency events
- Implement log rotation and archival

### Alerting Optimization
- Group related alerts to reduce noise
- Implement escalation policies
- Use inhibition rules to suppress dependent alerts

## Security Considerations

### Metrics Security
- Use RBAC for Prometheus access
- Secure Grafana with SSO
- Encrypt metrics in transit

### Log Security
- Sanitize sensitive data before logging
- Use secure transport for log shipping
- Implement log access controls

## Maintenance

### Regular Tasks
- Review and update alert thresholds
- Clean up old metrics and logs
- Update dashboard templates
- Test disaster recovery procedures

### Monitoring the Monitors
- Monitor Prometheus disk usage
- Check Fluentd buffer health
- Verify Azure Log Analytics ingestion
- Test alert delivery mechanisms