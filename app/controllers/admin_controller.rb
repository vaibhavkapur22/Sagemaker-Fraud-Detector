class AdminController < ApplicationController
  skip_before_action :verify_authenticity_token

  # POST /admin/seed
  # Seeds the database with sample transaction data
  def seed
    # Clear existing transactions
    Transaction.delete_all

    countries = %w[US UK CA DE FR IN AU BR JP MX NG RU CN]
    banks = %w[chase bank_of_america wells_fargo citibank capital_one us_bank pnc_bank td_bank hsbc barclays goldman_sachs morgan_stanley deutsche_bank credit_suisse santander bnp_paribas]
    sdk_types = %w[web ios android java python ruby]
    card_networks = %w[visa mastercard amex discover jcb]
    high_risk_countries = %w[NG RU CN]
    medium_risk_countries = %w[BR MX IN]

    transactions = []
    transaction_count = 10_000

    transaction_count.times do
      country = countries.sample
      bank = banks.sample
      sdk_type = sdk_types.sample
      card_network = card_networks.sample
      bin = "#{%w[4111 4000 4242 5100 5200 5300 3782 6011 3528].sample}#{rand(10..99)}"

      # Generate IP
      ip_address = if rand < 0.2
        ["10.#{rand(0..255)}.#{rand(0..255)}.#{rand(0..255)}",
         "192.168.#{rand(0..255)}.#{rand(0..255)}",
         "172.#{rand(16..31)}.#{rand(0..255)}.#{rand(0..255)}"].sample
      else
        "#{rand(1..223)}.#{rand(0..255)}.#{rand(0..255)}.#{rand(0..255)}"
      end

      # Calculate decline probability
      prob = 0.05
      prob += 0.25 if high_risk_countries.include?(country)
      prob += 0.10 if medium_risk_countries.include?(country)
      prob += 0.08 if sdk_type == 'web'
      prob -= 0.02 if %w[ios android].include?(sdk_type)
      prob += 0.15 if ip_address.start_with?('10.', '192.168.', '172.')
      prob += 0.05 if %w[santander bnp_paribas credit_suisse].include?(bank)
      prob = [[prob, 0.0].max, 0.6].min

      status = rand < prob ? 'declined' : 'approved'
      is_fraud = status == 'declined' && rand < 0.3
      processed_at = rand(30.days).seconds.ago

      transactions << {
        transaction_id: "txn_#{SecureRandom.hex(12)}",
        country: country,
        bank: bank,
        bin: bin,
        sdk_type: sdk_type,
        ip_address: ip_address,
        card_network: card_network,
        status: status,
        amount: (rand * 1000).round(2),
        currency: 'USD',
        fraud_score: rand.round(3),
        is_fraud: is_fraud,
        processed_at: processed_at,
        created_at: processed_at,
        updated_at: processed_at
      }

      # Batch insert
      if transactions.size >= 500
        Transaction.insert_all(transactions)
        transactions = []
      end
    end

    Transaction.insert_all(transactions) if transactions.any?

    # Sync to Redis
    sync_transactions_to_redis

    render json: {
      status: 'success',
      transactions_created: Transaction.count,
      approved: Transaction.approved.count,
      declined: Transaction.declined.count,
      decline_rate: (Transaction.declined.count.to_f / Transaction.count * 100).round(2)
    }
  end

  # POST /admin/sync_redis
  # Syncs transaction data to Redis for decline rate calculations
  def sync_redis
    count = sync_transactions_to_redis

    render json: {
      status: 'success',
      transactions_synced: count,
      redis_available: $redis&.ping == 'PONG'
    }
  end

  private

  def sync_transactions_to_redis
    return 0 unless $redis&.ping == 'PONG'

    calculator = DeclineRateCalculator.new
    transactions = Transaction.where('processed_at > ?', 15.days.ago)

    transactions.find_each.with_index do |txn, index|
      calculator.record_transaction(
        {
          country: txn.country,
          bank: txn.bank,
          bin: txn.bin,
          sdk_type: txn.sdk_type,
          ip_address: txn.ip_address,
          card_network: txn.card_network
        },
        status: txn.status
      )
    end

    transactions.count
  rescue => e
    Rails.logger.error("Redis sync error: #{e.message}")
    0
  end
end
