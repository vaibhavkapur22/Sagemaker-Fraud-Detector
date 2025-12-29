class SagemakerClient
  class PredictionError < StandardError; end

  DECLINE_RATE_FEATURES = %w[
    country_decline_rate_7d country_decline_rate_14d
    bank_decline_rate_7d bank_decline_rate_14d
    bin_decline_rate_7d bin_decline_rate_14d
    sdk_type_decline_rate_7d sdk_type_decline_rate_14d
    ip_decline_rate_7d ip_decline_rate_14d
    card_network_decline_rate_7d card_network_decline_rate_14d
  ].freeze

  def initialize
    @endpoint_name = ENV.fetch("SAGEMAKER_ENDPOINT_NAME", nil)
    @mock_mode = ENV.fetch("SAGEMAKER_MOCK_MODE", "true") == "true"
    @decline_rate_calculator = DeclineRateCalculator.new
  end

  def predict(features)
    debug_info = { steps: [], timings: {} }
    start_time = Time.current

    # Step 1: Fetch decline rates from Redis/RDS (real-time computation)
    redis_start = Time.current
    decline_rates = @decline_rate_calculator.calculate_rates(features)
    redis_available = @decline_rate_calculator.send(:redis_available?)
    debug_info[:timings][:decline_rates_ms] = ((Time.current - redis_start) * 1000).round(2)
    debug_info[:steps] << {
      service: redis_available ? "ElastiCache Redis" : "RDS PostgreSQL",
      action: "Fetch decline rates (7d & 14d windows)",
      status: "success",
      icon: redis_available ? "database" : "server",
      details: "#{decline_rates.keys.count} rate features computed"
    }
    debug_info[:decline_rates] = decline_rates
    debug_info[:redis_available] = redis_available

    # Merge original features with decline rates
    enriched_features = features.merge(decline_rates)

    # Step 2: Call SageMaker endpoint
    sagemaker_start = Time.current
    if @mock_mode
      result = mock_predict(enriched_features)
      debug_info[:steps] << {
        service: "SageMaker (Mock Mode)",
        action: "XGBoost inference",
        status: "success",
        icon: "cpu",
        details: "Mock prediction - demo mode"
      }
    else
      result = real_predict(enriched_features)
      debug_info[:steps] << {
        service: "SageMaker Endpoint",
        action: "XGBoost inference",
        status: "success",
        icon: "cpu",
        details: "Endpoint: #{@endpoint_name}"
      }
    end
    debug_info[:timings][:sagemaker_ms] = ((Time.current - sagemaker_start) * 1000).round(2)
    debug_info[:timings][:total_ms] = ((Time.current - start_time) * 1000).round(2)
    debug_info[:mock_mode] = @mock_mode
    debug_info[:endpoint_name] = @endpoint_name

    result.merge(debug: debug_info)
  rescue StandardError => e
    Rails.logger.error("SageMaker prediction error: #{e.message}")
    { score: nil, error: e.message, risk_level: "unknown", debug: debug_info }
  end

  private

  def real_predict(features)
    client = Aws::SageMakerRuntime::Client.new(
      region: ENV.fetch("AWS_REGION", "us-east-1"),
      credentials: Aws::Credentials.new(
        ENV.fetch("AWS_ACCESS_KEY_ID"),
        ENV.fetch("AWS_SECRET_ACCESS_KEY")
      ),
      ssl_verify_peer: false # Skip SSL verification for development
    )

    # Build CSV payload (SageMaker built-in XGBoost expects CSV)
    csv_payload = build_csv_payload(features)

    response = client.invoke_endpoint(
      endpoint_name: @endpoint_name,
      content_type: "text/csv",
      body: csv_payload
    )

    parse_csv_response(response.body.read)
  end

  def mock_predict(features)
    # Simulate realistic fraud scoring based on features
    base_score = 0.10

    # Country risk factor
    high_risk_countries = %w[NG RU CN]
    medium_risk_countries = %w[BR MX IN]
    if high_risk_countries.include?(features[:country])
      base_score += 0.25
    elsif medium_risk_countries.include?(features[:country])
      base_score += 0.12
    end

    # BIN risk (certain ranges are higher risk)
    bin_prefix = features[:bin].to_s[0..1]
    if %w[40 41].include?(bin_prefix) # Test BINs
      base_score += 0.08
    end

    # SDK type risk
    if features[:sdk_type] == "web"
      base_score += 0.08
    elsif %w[java python ruby].include?(features[:sdk_type])
      base_score += 0.04 # Server-side SDKs slightly higher risk
    elsif %w[ios android].include?(features[:sdk_type])
      base_score -= 0.02 # Native apps slightly safer
    end

    # Card network risk
    if features[:card_network] == "amex"
      base_score -= 0.03 # Amex has stricter verification
    end

    # Bank risk
    if %w[santander bnp_paribas].include?(features[:bank])
      base_score += 0.02 # European banks slightly different risk profile
    end

    # ============================================
    # Decline rate features influence (NEW)
    # Higher historical decline rates = higher risk
    # ============================================

    # 7-day decline rates have stronger influence (more recent)
    avg_7d_rate = calculate_average_decline_rate(features, '7d')
    avg_14d_rate = calculate_average_decline_rate(features, '14d')

    # Weight 7-day rates more heavily
    base_score += avg_7d_rate * 0.35
    base_score += avg_14d_rate * 0.20

    # Specific high-impact decline rates
    country_7d = features[:country_decline_rate_7d].to_f
    ip_7d = features[:ip_decline_rate_7d].to_f

    # High country decline rate is a strong signal
    base_score += 0.15 if country_7d > 0.25

    # High IP decline rate suggests suspicious IP
    base_score += 0.12 if ip_7d > 0.20

    # Add some randomness for demo purposes
    score = (base_score + rand(-0.05..0.05)).clamp(0.0, 1.0).round(3)

    build_result(score)
  end

  def calculate_average_decline_rate(features, window)
    rates = DECLINE_RATE_FEATURES.select { |f| f.end_with?(window) }.map do |feature|
      features[feature.to_sym].to_f
    end
    return 0.0 if rates.empty?

    rates.sum / rates.size
  end

  # Label encoders for categorical features (must match training)
  ENCODERS = {
    country: %w[AU BR CA CN DE FR IN JP MX NG RU UK US],
    bank: %w[bank_of_america barclays bnp_paribas capital_one chase citibank
             credit_suisse deutsche_bank goldman_sachs hsbc morgan_stanley
             pnc_bank santander td_bank us_bank wells_fargo],
    sdk_type: %w[android ios java python ruby web],
    card_network: %w[amex discover jcb mastercard visa]
  }.freeze

  def encode_categorical(value, encoder_key)
    encoder = ENCODERS[encoder_key]
    return -1 unless encoder

    index = encoder.index(value.to_s)
    index.nil? ? -1 : index
  end

  def build_csv_payload(features)
    # Encode categorical features
    country_enc = encode_categorical(features[:country], :country)
    bank_enc = encode_categorical(features[:bank], :bank)
    sdk_enc = encode_categorical(features[:sdk_type], :sdk_type)
    network_enc = encode_categorical(features[:card_network], :card_network)

    # Numeric features
    bin_numeric = features[:bin].to_s.to_i
    ip_parts = features[:ip_address].to_s.split(".")
    ip_first_octet = ip_parts.first&.to_i || 0
    ip_is_private = features[:ip_address].to_s.start_with?("10.", "192.168.", "172.") ? 1 : 0

    # Build feature array in exact order expected by model:
    # country, bank, bin, sdk_type, card_network, ip_first_octet, ip_is_private
    # + 12 decline rate features
    values = [
      country_enc,
      bank_enc,
      bin_numeric,
      sdk_enc,
      network_enc,
      ip_first_octet,
      ip_is_private,
      # Decline rate features (using ip_ not ip_address_ to match model)
      features[:country_decline_rate_7d] || 0.0,
      features[:country_decline_rate_14d] || 0.0,
      features[:bank_decline_rate_7d] || 0.0,
      features[:bank_decline_rate_14d] || 0.0,
      features[:bin_decline_rate_7d] || 0.0,
      features[:bin_decline_rate_14d] || 0.0,
      features[:sdk_type_decline_rate_7d] || 0.0,
      features[:sdk_type_decline_rate_14d] || 0.0,
      features[:ip_decline_rate_7d] || 0.0,
      features[:ip_decline_rate_14d] || 0.0,
      features[:card_network_decline_rate_7d] || 0.0,
      features[:card_network_decline_rate_14d] || 0.0
    ]

    values.join(",")
  end

  def parse_csv_response(body)
    # SageMaker XGBoost returns just the score as a float
    score = body.to_s.strip.to_f.round(3)
    build_result(score)
  end

  def build_result(score)
    {
      score: score,
      risk_level: calculate_risk_level(score),
      confidence: calculate_confidence(score),
      error: nil
    }
  end

  def calculate_risk_level(score)
    case score
    when 0.0..0.3
      "low"
    when 0.3..0.6
      "medium"
    when 0.6..0.8
      "high"
    else
      "critical"
    end
  end

  def calculate_confidence(score)
    # Confidence is higher when score is closer to extremes
    distance_from_center = (score - 0.5).abs
    ((0.5 + distance_from_center) * 100).round
  end
end
