module Webhooks
  # Controller for receiving transaction outcome webhooks from payment processors
  #
  # Endpoints:
  #   POST /webhooks/transactions - Record a new transaction outcome
  #   PATCH /webhooks/transactions/:id - Update an existing transaction's status
  #
  # Authentication:
  #   Uses webhook signature verification via X-Webhook-Signature header
  #
  class TransactionsController < ApplicationController
    skip_before_action :verify_authenticity_token
    before_action :verify_webhook_signature, if: -> { webhook_signature_required? }

    # POST /webhooks/transactions
    # Record a new transaction with its outcome
    #
    # Request body:
    # {
    #   "transaction": {
    #     "transaction_id": "optional_external_id",
    #     "country": "US",
    #     "bank": "chase",
    #     "bin": "411111",
    #     "sdk_type": "ios",
    #     "ip_address": "192.168.1.1",
    #     "card_network": "visa",
    #     "status": "approved",
    #     "amount": 99.99,
    #     "fraud_score": 0.15
    #   }
    # }
    def create
      recorder = TransactionRecorder.new

      transaction = recorder.record(
        transaction_params.to_h.symbolize_keys,
        status: params.dig(:transaction, :status) || 'pending',
        fraud_score: params.dig(:transaction, :fraud_score)&.to_f,
        amount: params.dig(:transaction, :amount)&.to_f
      )

      render json: {
        success: true,
        transaction_id: transaction.transaction_id,
        status: transaction.status
      }, status: :created
    rescue ActiveRecord::RecordInvalid => e
      render json: { success: false, error: e.message }, status: :unprocessable_entity
    rescue StandardError => e
      Rails.logger.error("Webhook error: #{e.message}")
      render json: { success: false, error: 'Internal server error' }, status: :internal_server_error
    end

    # PATCH /webhooks/transactions/:id
    # Update an existing transaction's status
    #
    # Request body:
    # {
    #   "status": "declined"
    # }
    def update
      recorder = TransactionRecorder.new
      transaction = recorder.update_status(params[:id], status: params[:status])

      if transaction
        render json: {
          success: true,
          transaction_id: transaction.transaction_id,
          status: transaction.status
        }
      else
        render json: { success: false, error: 'Transaction not found' }, status: :not_found
      end
    rescue StandardError => e
      Rails.logger.error("Webhook update error: #{e.message}")
      render json: { success: false, error: 'Internal server error' }, status: :internal_server_error
    end

    # POST /webhooks/transactions/batch
    # Record multiple transactions at once
    #
    # Request body:
    # {
    #   "transactions": [
    #     { "country": "US", "bank": "chase", ... , "status": "approved" },
    #     { "country": "NG", "bank": "hsbc", ... , "status": "declined" }
    #   ]
    # }
    def batch
      recorder = TransactionRecorder.new

      transactions = params[:transactions].map do |txn|
        {
          features: txn.permit(:country, :bank, :bin, :sdk_type, :ip_address, :card_network).to_h.symbolize_keys,
          status: txn[:status],
          fraud_score: txn[:fraud_score]&.to_f,
          amount: txn[:amount]&.to_f
        }
      end

      result = recorder.record_batch(transactions)

      render json: {
        success: true,
        recorded: result[:success],
        failed: result[:failed]
      }, status: :created
    rescue StandardError => e
      Rails.logger.error("Webhook batch error: #{e.message}")
      render json: { success: false, error: 'Internal server error' }, status: :internal_server_error
    end

    private

    def transaction_params
      params.require(:transaction).permit(
        :country, :bank, :bin, :sdk_type, :ip_address, :card_network
      )
    end

    def webhook_signature_required?
      ENV.fetch('WEBHOOK_SIGNATURE_REQUIRED', 'false') == 'true'
    end

    def verify_webhook_signature
      signature = request.headers['X-Webhook-Signature']
      secret = ENV.fetch('WEBHOOK_SECRET', nil)

      return if secret.blank?

      expected_signature = compute_signature(request.raw_post, secret)

      unless ActiveSupport::SecurityUtils.secure_compare(signature.to_s, expected_signature)
        render json: { success: false, error: 'Invalid signature' }, status: :unauthorized
      end
    end

    def compute_signature(payload, secret)
      OpenSSL::HMAC.hexdigest('SHA256', secret, payload)
    end
  end
end
