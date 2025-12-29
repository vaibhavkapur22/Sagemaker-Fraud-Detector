# Redis configuration for decline rate tracking
# Uses ElastiCache in production, local Redis in development

require 'redis'

redis_config = {
  url: ENV.fetch('REDIS_URL', 'redis://localhost:6379/0'),
  connect_timeout: 2,
  read_timeout: 1,
  write_timeout: 1
}

# Production configuration for AWS ElastiCache
if Rails.env.production?
  redis_url = ENV.fetch('ELASTICACHE_URL', ENV.fetch('REDIS_URL', 'redis://localhost:6379/0'))

  redis_config.merge!(
    url: redis_url,
    ssl: ENV.fetch('REDIS_SSL', 'true') == 'true',
    ssl_params: { verify_mode: OpenSSL::SSL::VERIFY_NONE }
  )

  # Enable cluster mode if configured
  if ENV.fetch('REDIS_CLUSTER_MODE', 'false') == 'true'
    redis_config[:cluster] = ENV.fetch('REDIS_CLUSTER_NODES', '').split(',')
  end
end

REDIS_CONFIG = redis_config.freeze

# Initialize Redis connection
begin
  $redis = Redis.new(REDIS_CONFIG)
  # Test connection
  $redis.ping
  Rails.logger.info "Redis connected successfully to #{REDIS_CONFIG[:url]}"
rescue Redis::CannotConnectError => e
  Rails.logger.warn "Redis connection failed: #{e.message}. Decline rate features will use database fallback."
  $redis = nil
rescue StandardError => e
  Rails.logger.error "Redis initialization error: #{e.message}"
  $redis = nil
end

# Helper method to check if Redis is available
def redis_available?
  $redis&.ping == 'PONG'
rescue StandardError
  false
end
