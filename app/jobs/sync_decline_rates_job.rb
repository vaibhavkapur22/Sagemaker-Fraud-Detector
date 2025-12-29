# Job to sync transaction data from PostgreSQL to Redis for decline rate calculations
# Run this job periodically or on-demand to ensure Redis has accurate decline rate data
#
# Usage:
#   SyncDeclineRatesJob.perform_now                    # Sync last 15 days (default)
#   SyncDeclineRatesJob.perform_now(lookback_days: 30) # Sync last 30 days
#   SyncDeclineRatesJob.perform_later                  # Queue for background processing
#
class SyncDeclineRatesJob < ApplicationJob
  queue_as :default

  # Batch size for processing transactions
  BATCH_SIZE = 1000

  def perform(lookback_days: 15)
    Rails.logger.info "Starting decline rate sync for last #{lookback_days} days"

    calculator = DeclineRateCalculator.new

    # Clear existing data to rebuild from scratch
    calculator.clear_all
    Rails.logger.info "Cleared existing Redis decline rate data"

    processed_count = 0
    start_time = Time.current

    # Process transactions in batches
    Transaction
      .where('processed_at >= ?', lookback_days.days.ago)
      .where(status: %w[approved declined])
      .find_in_batches(batch_size: BATCH_SIZE) do |batch|

      # Transform to format expected by batch_record
      transactions = batch.map do |txn|
        {
          country: txn.country,
          bank: txn.bank,
          bin: txn.bin,
          sdk_type: txn.sdk_type,
          ip_address: txn.ip_address,
          card_network: txn.card_network,
          status: txn.status,
          processed_at: txn.processed_at
        }
      end

      calculator.batch_record(transactions)
      processed_count += batch.size

      Rails.logger.info "Synced #{processed_count} transactions..." if (processed_count % 5000).zero?
    end

    elapsed = Time.current - start_time
    Rails.logger.info "Decline rate sync complete: #{processed_count} transactions in #{elapsed.round(2)}s"

    { processed: processed_count, elapsed_seconds: elapsed.round(2) }
  end
end
