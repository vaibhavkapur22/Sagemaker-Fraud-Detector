# Client for interacting with SageMaker Feature Store
# Used for online feature lookup during inference
class FeatureStoreClient
  DECLINE_RATE_FEATURES = %w[
    country_decline_rate_7d country_decline_rate_14d
    bank_decline_rate_7d bank_decline_rate_14d
    bin_decline_rate_7d bin_decline_rate_14d
    sdk_type_decline_rate_7d sdk_type_decline_rate_14d
    ip_decline_rate_7d ip_decline_rate_14d
    card_network_decline_rate_7d card_network_decline_rate_14d
  ].freeze

  def initialize
    @feature_group_name = ENV.fetch('FEATURE_GROUP_NAME', 'fraud-detection-decline-rates')
    @region = ENV.fetch('AWS_REGION', 'us-east-1')
    @client = build_client
  end

  # Fetch a record from the online store
  # @param record_id [String] The record identifier
  # @return [Hash, nil] Record data or nil if not found
  def get_record(record_id)
    return nil unless @client

    response = @client.get_record(
      feature_group_name: @feature_group_name,
      record_identifier_value_as_string: record_id
    )

    return nil unless response.record

    response.record.each_with_object({}) do |feature, hash|
      hash[feature.feature_name.to_sym] = parse_value(feature.value_as_string)
    end
  rescue Aws::SageMakerFeatureStoreRuntime::Errors::ResourceNotFound
    nil
  rescue StandardError => e
    Rails.logger.error("Feature Store get_record error: #{e.message}")
    nil
  end

  # Fetch only decline rate features for a record
  # @param record_id [String] The record identifier
  # @return [Hash] Decline rate features (returns zeros if not found)
  def get_decline_rates(record_id)
    record = get_record(record_id)

    if record.nil?
      # Return zeros if record not found
      return DECLINE_RATE_FEATURES.each_with_object({}) do |feature, hash|
        hash[feature.to_sym] = 0.0
      end
    end

    DECLINE_RATE_FEATURES.each_with_object({}) do |feature, hash|
      hash[feature.to_sym] = record[feature.to_sym].to_f
    end
  end

  # Write a record to the feature store
  # @param record [Hash] Record data with all features
  # @return [Boolean] Success status
  def put_record(record)
    return false unless @client

    feature_records = record.map do |key, value|
      { feature_name: key.to_s, value_as_string: value.to_s }
    end

    @client.put_record(
      feature_group_name: @feature_group_name,
      record: feature_records
    )

    true
  rescue StandardError => e
    Rails.logger.error("Feature Store put_record error: #{e.message}")
    false
  end

  # Build a complete record for the feature store
  # @param record_id [String] Unique identifier
  # @param features [Hash] Original transaction features
  # @param decline_rates [Hash] Computed decline rate features
  # @param is_fraud [Boolean] Whether the transaction is fraud
  # @param status [String] Transaction status
  # @return [Hash] Complete record ready for put_record
  def build_record(record_id:, features:, decline_rates:, is_fraud: false, status: 'pending')
    record = {
      record_id: record_id,
      event_time: Time.current.to_f,
      is_fraud: is_fraud ? 1 : 0,
      status: status
    }

    # Add original features
    %i[country bank bin sdk_type ip_address card_network].each do |key|
      record[key] = features[key] || ''
    end

    # Add decline rate features
    DECLINE_RATE_FEATURES.each do |feature|
      record[feature.to_sym] = decline_rates[feature.to_sym] || 0.0
    end

    record
  end

  # Check if the Feature Store client is available
  # @return [Boolean]
  def available?
    @client.present?
  end

  private

  def build_client
    Aws::SageMakerFeatureStoreRuntime::Client.new(
      region: @region,
      credentials: Aws::Credentials.new(
        ENV.fetch('AWS_ACCESS_KEY_ID', nil),
        ENV.fetch('AWS_SECRET_ACCESS_KEY', nil)
      )
    )
  rescue StandardError => e
    Rails.logger.warn("Feature Store client initialization failed: #{e.message}")
    nil
  end

  def parse_value(value)
    return nil if value.nil?

    # Try to parse as number
    if value.include?('.')
      Float(value)
    else
      Integer(value)
    end
  rescue ArgumentError
    value
  end
end
