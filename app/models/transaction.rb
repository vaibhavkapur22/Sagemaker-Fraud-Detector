class Transaction < ApplicationRecord
  STATUSES = %w[approved declined pending].freeze
  DIMENSIONS = %w[country bank bin sdk_type ip_address card_network].freeze

  validates :transaction_id, presence: true, uniqueness: true
  validates :country, :bank, :bin, :sdk_type, :ip_address, :card_network, presence: true
  validates :status, presence: true, inclusion: { in: STATUSES }
  validates :processed_at, presence: true

  scope :approved, -> { where(status: 'approved') }
  scope :declined, -> { where(status: 'declined') }
  scope :pending, -> { where(status: 'pending') }
  scope :in_window, ->(days) { where('processed_at >= ?', days.days.ago) }
  scope :before, ->(time) { where('processed_at < ?', time) }

  # Calculate decline rate for a specific dimension and value within a time window
  # @param dimension [String] one of: country, bank, bin, sdk_type, ip_address, card_network
  # @param value [String] the value to filter by (e.g., 'US' for country)
  # @param days [Integer] number of days to look back
  # @param before_time [Time] optional, calculate rate before this time (for point-in-time accuracy)
  # @return [Float] decline rate as a decimal (0.0 to 1.0)
  def self.decline_rate_for(dimension, value, days:, before_time: nil)
    raise ArgumentError, "Invalid dimension: #{dimension}" unless DIMENSIONS.include?(dimension.to_s)

    scope = where(dimension => value)

    if before_time
      scope = scope.where('processed_at >= ? AND processed_at < ?', before_time - days.days, before_time)
    else
      scope = scope.in_window(days)
    end

    # Only count completed transactions (approved or declined)
    scope = scope.where(status: %w[approved declined])

    total = scope.count
    return 0.0 if total.zero?

    declined_count = scope.declined.count
    (declined_count.to_f / total).round(4)
  end

  # Calculate all decline rates for a given set of features
  # @param features [Hash] hash with keys: country, bank, bin, sdk_type, ip_address, card_network
  # @param windows [Array<Integer>] array of day windows (default: [7, 14])
  # @param before_time [Time] optional, calculate rates before this time
  # @return [Hash] hash with decline rate feature names and values
  def self.decline_rates_for_features(features, windows: [7, 14], before_time: nil)
    rates = {}

    DIMENSIONS.each do |dimension|
      value = features[dimension.to_sym] || features[dimension]
      next if value.blank?

      windows.each do |days|
        feature_name = "#{dimension}_decline_rate_#{days}d"
        rates[feature_name.to_sym] = decline_rate_for(dimension, value, days: days, before_time: before_time)
      end
    end

    rates
  end

  # Convert to features hash for ML model
  def to_features
    {
      country: country,
      bank: bank,
      bin: bin,
      sdk_type: sdk_type,
      ip_address: ip_address,
      card_network: card_network
    }
  end
end
