Rails.application.routes.draw do
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Reveal health status on /up that returns 200 if the app boots with no exceptions, otherwise 500.
  # Can be used by load balancers and uptime monitors to verify that the app is live.
  get "up" => "rails/health#show", as: :rails_health_check

  # Fraud prediction routes
  resources :predictions, only: [:new, :create]

  # Transaction management routes
  resources :transactions, only: [:index, :show]

  # Webhooks for transaction outcomes from payment processors
  namespace :webhooks do
    resources :transactions, only: [:create, :update] do
      collection do
        post :batch
      end
    end
  end

  # Admin/setup routes (should be protected in production)
  post "admin/seed" => "admin#seed"
  post "admin/sync_redis" => "admin#sync_redis"

  # Defines the root path route ("/")
  root "predictions#new"
end
