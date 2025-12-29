# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[7.1].define(version: 2024_12_28_000001) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "plpgsql"

  create_table "transactions", force: :cascade do |t|
    t.string "transaction_id", null: false
    t.string "country", null: false
    t.string "bank", null: false
    t.string "bin", null: false
    t.string "sdk_type", null: false
    t.string "ip_address", null: false
    t.string "card_network", null: false
    t.string "status", default: "pending", null: false
    t.decimal "amount", precision: 10, scale: 2
    t.string "currency", default: "USD"
    t.float "fraud_score"
    t.boolean "is_fraud", default: false
    t.datetime "processed_at", null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["bank", "processed_at"], name: "index_transactions_on_bank_and_processed_at"
    t.index ["bin", "processed_at"], name: "index_transactions_on_bin_and_processed_at"
    t.index ["card_network", "processed_at"], name: "index_transactions_on_card_network_and_processed_at"
    t.index ["country", "processed_at"], name: "index_transactions_on_country_and_processed_at"
    t.index ["ip_address", "processed_at"], name: "index_transactions_on_ip_address_and_processed_at"
    t.index ["processed_at"], name: "index_transactions_on_processed_at"
    t.index ["sdk_type", "processed_at"], name: "index_transactions_on_sdk_type_and_processed_at"
    t.index ["status"], name: "index_transactions_on_status"
    t.index ["transaction_id"], name: "index_transactions_on_transaction_id", unique: true
  end

end
