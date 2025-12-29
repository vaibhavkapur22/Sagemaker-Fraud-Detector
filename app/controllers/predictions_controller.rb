class PredictionsController < ApplicationController
  def new
    @prediction = FraudPrediction.new
    @result = nil
  end

  def create
    @prediction = FraudPrediction.new(prediction_params)

    if @prediction.valid?
      @result = SagemakerClient.new.predict(@prediction.to_features)
    end

    respond_to do |format|
      format.turbo_stream
      format.html { render :new }
      format.json { render json: @result || { error: @prediction.errors.full_messages } }
    end
  end

  private

  def prediction_params
    params.require(:fraud_prediction).permit(
      :country, :bank, :bin, :sdk_type, :ip_address, :card_network
    )
  end
end
