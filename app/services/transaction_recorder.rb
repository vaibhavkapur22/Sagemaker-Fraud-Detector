# Service for recording transaction outcomes
# Updates both PostgreSQL (for historical analysis) and Redis (for real-time decline rates)
class TransactionRecorder
  def initialize
    @decline_rate_calculator = DeclineRateCalculator.new
  end

  # Record a new transaction with its outcome
  # @param features [Hash] transaction features (country, bank, bin, etc.)
  # @param status [String] 'approved' or 'declined'
  # @param fraud_score [Float] optional fraud score from model
  # @param amount [Float] optional transaction amount
  # @return [Transaction] the created transaction record
  def record(features, status:, fraud_score: nil, amount: nil)
    transaction = create_transaction(features, status, fraud_score, amount)

    # Update Redis counters for real-time decline rate calculation
    update_redis_counters(features, status)

    transaction
  end

  # Record multiple transactions in batch
  # @param transactions [Array<Hash>] array of { features:, status:, fraud_score:, amount: }
  # @return [Hash] { success: count, failed: count }
  def record_batch(transactions)
    success_count = 0
    failed_count = 0

    transactions.each do |txn|
      record(
        txn[:features],
        status: txn[:status],
        fraud_score: txn[:fraud_score],
        amount: txn[:amount]
      )
      success_count += 1
    rescue StandardError => e
      Rails.logger.error("Failed to record transaction: #{e.message}")
      failed_count += 1
    end

    { success: success_count, failed: failed_count }
  end

  # Update an existing transaction's status (e.g., pending -> approved)
  # @param transaction_id [String] the transaction ID
  # @param status [String] new status
  # @return [Transaction, nil] the updated transaction or nil if not found
  def update_status(transaction_id, status:)
    transaction = Transaction.find_by(transaction_id: transaction_id)
    return nil unless transaction

    old_status = transaction.status
    transaction.update!(status: status)

    # If status changed to approved/declined, update Redis
    if %w[approved declined].include?(status) && old_status == 'pending'
      update_redis_counters(transaction.to_features, status)
    end

    transaction
  end

  private

  def create_transaction(features, status, fraud_score, amount)
    Transaction.create!(
      transaction_id: generate_transaction_id,
      country: features[:country],
      bank: features[:bank],
      bin: features[:bin],
      sdk_type: features[:sdk_type],
      ip_address: features[:ip_address],
      card_network: features[:card_network],
      status: status,
      fraud_score: fraud_score,
      amount: amount,
      is_fraud: fraud_score.present? && fraud_score > 0.5,
      processed_at: Time.current
    )
  end

  def update_redis_counters(features, status)
    @decline_rate_calculator.record_transaction(features, status: status)
  rescue StandardError => e
    Rails.logger.error("Failed to update Redis counters: #{e.message}")
  end

  def generate_transaction_id
    "txn_#{SecureRandom.hex(12)}"
  end
end
