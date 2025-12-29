# This file should ensure the existence of records required to run the application in every environment (production,
# development, test). The code here should be idempotent so that it can be executed at any point in every environment.
# The data can then be loaded with the bin/rails db:seed command (or created alongside the database with db:setup).

# Generate sample transaction data for decline rate feature testing
# This creates ~10,000 transactions over the past 30 days with realistic decline patterns

puts "Seeding transaction data..."

COUNTRIES = %w[US UK CA DE FR IN AU BR JP MX NG RU CN].freeze
BANKS = %w[chase bank_of_america wells_fargo citibank capital_one us_bank pnc_bank td_bank hsbc barclays goldman_sachs morgan_stanley deutsche_bank credit_suisse santander bnp_paribas].freeze
SDK_TYPES = %w[web ios android java python ruby].freeze
CARD_NETWORKS = %w[visa mastercard amex discover jcb].freeze

# Risk profiles for different dimensions (affects decline probability)
HIGH_RISK_COUNTRIES = %w[NG RU CN].freeze
MEDIUM_RISK_COUNTRIES = %w[BR MX IN].freeze
HIGH_RISK_SDKS = %w[web].freeze

def generate_bin
  # Generate realistic BIN prefixes
  prefixes = %w[4111 4000 4242 5100 5200 5300 3782 6011 3528]
  "#{prefixes.sample}#{rand(10..99)}"
end

def generate_ip
  # Mix of public and private IPs
  if rand < 0.2
    # Private IP (higher fraud risk indicator)
    ["10.#{rand(0..255)}.#{rand(0..255)}.#{rand(0..255)}",
     "192.168.#{rand(0..255)}.#{rand(0..255)}",
     "172.#{rand(16..31)}.#{rand(0..255)}.#{rand(0..255)}"].sample
  else
    # Public IP
    "#{rand(1..223)}.#{rand(0..255)}.#{rand(0..255)}.#{rand(0..255)}"
  end
end

def calculate_decline_probability(country:, sdk_type:, ip_address:, bank:)
  # Base decline rate of 5%
  prob = 0.05

  # Country risk adjustment
  if HIGH_RISK_COUNTRIES.include?(country)
    prob += 0.25
  elsif MEDIUM_RISK_COUNTRIES.include?(country)
    prob += 0.10
  end

  # SDK type risk adjustment
  prob += 0.08 if HIGH_RISK_SDKS.include?(sdk_type)
  prob -= 0.02 if %w[ios android].include?(sdk_type)

  # Private IP increases risk
  prob += 0.15 if ip_address.start_with?('10.', '192.168.', '172.')

  # Some banks have higher decline rates
  prob += 0.05 if %w[santander bnp_paribas credit_suisse].include?(bank)

  # Cap probability
  [[prob, 0.0].max, 0.6].min
end

# Clear existing transactions
Transaction.delete_all
puts "Cleared existing transactions"

# Generate transactions over the past 30 days
transaction_count = 10_000
batch_size = 500
transactions = []

transaction_count.times do |i|
  country = COUNTRIES.sample
  bank = BANKS.sample
  sdk_type = SDK_TYPES.sample
  ip_address = generate_ip
  card_network = CARD_NETWORKS.sample
  bin = generate_bin

  decline_prob = calculate_decline_probability(
    country: country,
    sdk_type: sdk_type,
    ip_address: ip_address,
    bank: bank
  )

  status = rand < decline_prob ? 'declined' : 'approved'
  is_fraud = status == 'declined' && rand < 0.3 # 30% of declines are actual fraud

  # Distribute transactions over the past 30 days
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

  # Batch insert for performance
  if transactions.size >= batch_size
    Transaction.insert_all(transactions)
    transactions = []
    print "." if (i % 1000).zero?
  end
end

# Insert remaining transactions
Transaction.insert_all(transactions) if transactions.any?

puts "\nCreated #{Transaction.count} transactions"

# Print summary statistics
puts "\nTransaction Summary:"
puts "  Total: #{Transaction.count}"
puts "  Approved: #{Transaction.approved.count}"
puts "  Declined: #{Transaction.declined.count}"
puts "  Overall decline rate: #{(Transaction.declined.count.to_f / Transaction.count * 100).round(2)}%"

puts "\nDecline rates by country (last 7 days):"
COUNTRIES.each do |country|
  rate = Transaction.decline_rate_for('country', country, days: 7)
  count = Transaction.where(country: country).in_window(7).count
  puts "  #{country}: #{(rate * 100).round(2)}% (#{count} transactions)"
end

puts "\nDecline rates by SDK type (last 7 days):"
SDK_TYPES.each do |sdk|
  rate = Transaction.decline_rate_for('sdk_type', sdk, days: 7)
  count = Transaction.where(sdk_type: sdk).in_window(7).count
  puts "  #{sdk}: #{(rate * 100).round(2)}% (#{count} transactions)"
end

puts "\nSeeding complete!"
