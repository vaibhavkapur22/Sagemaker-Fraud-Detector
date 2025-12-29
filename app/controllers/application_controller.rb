class ApplicationController < ActionController::Base
  # Use null_session strategy to avoid CSRF issues for API requests
  # This resets the session instead of raising an exception
  protect_from_forgery with: :null_session
end
