# Service for calculating decline rates using Redis for real-time computation
# Falls back to database queries if Redis is unavailable
class DeclineRateCalculator
  DIMENSIONS = %w[country bank bin sdk_type ip_address card_network].freeze
  WINDOWS = [7, 14].freeze
  KEY_PREFIX = 'decline'.freeze
  TTL_DAYS = 15

  def initialize(redis: $redis)
    @redis = redis
  end

  # Record a transaction outcome to Redis
  # @param attributes [Hash] transaction attributes with dimension values
  # @param status [String] 'approved' or 'declined'
  def record_transaction(attributes, status:)
    return unless redis_available?
    return unless %w[approved declined].include?(status)

    date = Date.current.to_s

    DIMENSIONS.each do |dimension|
      value = extract_value(attributes, dimension)
      next if value.blank?

      key = build_key(dimension, value, date)

      @redis.pipelined do |pipeline|
        pipeline.hincrby(key, 'total', 1)
        pipeline.hincrby(key, 'declined', 1) if status == 'declined'
        pipeline.expire(key, TTL_DAYS.days.to_i)
      end
    end
  rescue Redis::BaseError => e
    Rails.logger.error("Redis error recording transaction: #{e.message}")
  end

  # Calculate all decline rates for a transaction's features
  # @param attributes [Hash] transaction attributes with dimension values
  # @return [Hash] hash with decline rate feature names and values
  def calculate_rates(attributes)
    rates = {}

    DIMENSIONS.each do |dimension|
      value = extract_value(attributes, dimension)

      WINDOWS.each do |days|
        feature_name = "#{dimension}_decline_rate_#{days}d"
        rates[feature_name.to_sym] = calculate_rate(dimension, value, days)
      end
    end

    rates
  end

  # Calculate decline rate for a specific dimension/value/window
  # @param dimension [String] one of DIMENSIONS
  # @param value [String] the dimension value
  # @param days [Integer] number of days to look back
  # @return [Float] decline rate as decimal (0.0 to 1.0)
  def calculate_rate(dimension, value, days)
    return 0.0 if value.blank?

    if redis_available?
      calculate_rate_from_redis(dimension, value, days)
    else
      calculate_rate_from_database(dimension, value, days)
    end
  end

  # Batch record multiple transactions (for syncing from database)
  # @param transactions [Array<Hash>] array of transaction hashes with :attributes and :status
  def batch_record(transactions)
    return unless redis_available?

    @redis.pipelined do |pipeline|
      transactions.each do |txn|
        date = txn[:processed_at]&.to_date&.to_s || Date.current.to_s
        status = txn[:status]
        next unless %w[approved declined].include?(status)

        DIMENSIONS.each do |dimension|
          value = extract_value(txn, dimension)
          next if value.blank?

          key = build_key(dimension, value, date)
          pipeline.hincrby(key, 'total', 1)
          pipeline.hincrby(key, 'declined', 1) if status == 'declined'
          pipeline.expire(key, TTL_DAYS.days.to_i)
        end
      end
    end
  rescue Redis::BaseError => e
    Rails.logger.error("Redis error in batch_record: #{e.message}")
  end

  # Clear all decline rate data from Redis (useful for testing)
  def clear_all
    return unless redis_available?

    keys = @redis.keys("#{KEY_PREFIX}:*")
    @redis.del(*keys) if keys.any?
  end

  private

  def redis_available?
    @redis&.ping == 'PONG'
  rescue StandardError
    false
  end

  def extract_value(attributes, dimension)
    attributes[dimension.to_sym] || attributes[dimension.to_s] || attributes[dimension]
  end

  def build_key(dimension, value, date)
    "#{KEY_PREFIX}:#{dimension}:#{value}:#{date}"
  end

  def calculate_rate_from_redis(dimension, value, days)
    total = 0
    declined = 0

    # Collect keys for all days in the window
    keys = (0...days).map do |offset|
      date = (Date.current - offset.days).to_s
      build_key(dimension, value, date)
    end

    # Batch fetch all data
    results = @redis.pipelined do |pipeline|
      keys.each { |key| pipeline.hgetall(key) }
    end

    results.each do |data|
      total += data['total'].to_i
      declined += data['declined'].to_i
    end

    return 0.0 if total.zero?
    (declined.to_f / total).round(4)
  rescue Redis::BaseError => e
    Rails.logger.error("Redis error calculating rate: #{e.message}")
    calculate_rate_from_database(dimension, value, days)
  end

  def calculate_rate_from_database(dimension, value, days)
    Transaction.decline_rate_for(dimension, value, days: days)
  rescue StandardError => e
    Rails.logger.error("Database error calculating rate: #{e.message}")
    0.0
  end
end
