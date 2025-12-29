# Form object for fraud prediction inputs (not persisted to database)
class FraudPrediction
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :country, :string
  attribute :bank, :string
  attribute :bin, :string
  attribute :sdk_type, :string
  attribute :ip_address, :string
  attribute :card_network, :string

  validates :country, presence: true
  validates :bank, presence: true
  validates :bin, presence: true, format: { with: /\A\d{6}\z/, message: "must be exactly 6 digits" }
  validates :sdk_type, presence: true
  validates :ip_address, presence: true, format: { with: /\A(\d{1,3}\.){3}\d{1,3}\z/, message: "must be a valid IPv4 address" }
  validates :card_network, presence: true

  COUNTRIES = [
    ["United States", "US"],
    ["United Kingdom", "UK"],
    ["Canada", "CA"],
    ["Germany", "DE"],
    ["France", "FR"],
    ["India", "IN"],
    ["Australia", "AU"],
    ["Brazil", "BR"],
    ["Japan", "JP"],
    ["Mexico", "MX"],
    ["Nigeria", "NG"],
    ["Russia", "RU"],
    ["China", "CN"]
  ].freeze

  BANKS = [
    ["Chase", "chase"],
    ["Bank of America", "bank_of_america"],
    ["Wells Fargo", "wells_fargo"],
    ["Citibank", "citibank"],
    ["Capital One", "capital_one"],
    ["US Bank", "us_bank"],
    ["PNC Bank", "pnc_bank"],
    ["TD Bank", "td_bank"],
    ["HSBC", "hsbc"],
    ["Barclays", "barclays"],
    ["Goldman Sachs", "goldman_sachs"],
    ["Morgan Stanley", "morgan_stanley"],
    ["Deutsche Bank", "deutsche_bank"],
    ["Credit Suisse", "credit_suisse"],
    ["Santander", "santander"],
    ["BNP Paribas", "bnp_paribas"]
  ].freeze

  SDK_TYPES = [
    ["Web", "web"],
    ["iOS", "ios"],
    ["Android", "android"],
    ["Java", "java"],
    ["Python", "python"],
    ["Ruby", "ruby"]
  ].freeze

  CARD_NETWORKS = [
    ["Visa", "visa"],
    ["Mastercard", "mastercard"],
    ["American Express", "amex"],
    ["Discover", "discover"],
    ["JCB", "jcb"]
  ].freeze

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
