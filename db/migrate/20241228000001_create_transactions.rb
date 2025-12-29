class CreateTransactions < ActiveRecord::Migration[7.1]
  def change
    create_table :transactions do |t|
      t.string :transaction_id, null: false
      t.string :country, null: false
      t.string :bank, null: false
      t.string :bin, null: false
      t.string :sdk_type, null: false
      t.string :ip_address, null: false
      t.string :card_network, null: false
      t.string :status, null: false, default: 'pending'
      t.decimal :amount, precision: 10, scale: 2
      t.string :currency, default: 'USD'
      t.float :fraud_score
      t.boolean :is_fraud, default: false
      t.datetime :processed_at, null: false

      t.timestamps
    end

    add_index :transactions, :transaction_id, unique: true
    add_index :transactions, [:country, :processed_at]
    add_index :transactions, [:bank, :processed_at]
    add_index :transactions, [:bin, :processed_at]
    add_index :transactions, [:sdk_type, :processed_at]
    add_index :transactions, [:ip_address, :processed_at]
    add_index :transactions, [:card_network, :processed_at]
    add_index :transactions, :processed_at
    add_index :transactions, :status
  end
end
