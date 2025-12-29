# SageMaker Fraud Detection Model

This directory contains scripts for training and deploying a fraud detection model to AWS SageMaker.

## Prerequisites

1. **Python 3.8+** with pip
2. **AWS Account** with SageMaker access
3. **AWS CLI** configured with credentials
4. **IAM Role** for SageMaker with appropriate permissions

## Setup

### 1. Install Dependencies

```bash
cd sagemaker
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Set your AWS credentials as environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
```

### 3. Create SageMaker Execution Role

Create an IAM role with the following policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scoped to your bucket)

## Usage

### Step 1: Generate Training Data

```bash
python generate_sample_data.py
```

This creates synthetic fraud detection data in `data/train.csv` and `data/test.csv`.

### Step 2: Train Model Locally (Optional)

For local testing before deploying to SageMaker:

```bash
python train_model.py
```

This trains an XGBoost model locally and saves it to `model/`.

### Step 3: Deploy to SageMaker

```bash
python deploy_to_sagemaker.py \
    --region us-east-1 \
    --endpoint-name fraud-detector-endpoint \
    --instance-type ml.m5.large \
    --deploy-instance-type ml.t2.medium
```

#### Options:
- `--region`: AWS region (default: us-east-1)
- `--bucket`: S3 bucket name (uses SageMaker default bucket if not specified)
- `--endpoint-name`: Name for the deployed endpoint
- `--instance-type`: Training instance type
- `--deploy-instance-type`: Endpoint instance type
- `--skip-training`: Skip training, deploy existing model

## Model Features

The model uses the following features:

| Feature | Type | Description |
|---------|------|-------------|
| country | Categorical | Transaction origin country code |
| bank | Categorical | Issuing bank name |
| bin | Numeric | First 6 digits of card number |
| sdk_type | Categorical | Platform (ios, android, web, etc.) |
| ip_address | Derived | Extracts first octet and private flag |
| card_network | Categorical | Payment network (visa, mastercard, etc.) |

## API Format

### Request

```json
{
  "instances": [
    {
      "country": "US",
      "bank": "Chase",
      "bin": "411111",
      "sdk_type": "ios",
      "ip_address": "192.168.1.1",
      "card_network": "visa"
    }
  ]
}
```

### Response

```json
{
  "predictions": [
    {
      "score": 0.15,
      "prediction": "legitimate",
      "confidence": 70.0
    }
  ]
}
```

## Cost Considerations

- **Training**: ml.m5.large costs ~$0.115/hour
- **Endpoint**: ml.t2.medium costs ~$0.056/hour (~$40/month if always on)
- **Storage**: S3 costs for training data and model artifacts

To minimize costs:
1. Use spot instances for training (add `use_spot_instances=True`)
2. Delete endpoint when not in use
3. Use smaller instance types for testing

## Cleanup

Delete the endpoint when not needed:

```bash
aws sagemaker delete-endpoint --endpoint-name fraud-detector-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name fraud-detector-endpoint
```

## Troubleshooting

### "SAGEMAKER_ROLE_ARN not set"
Set the IAM role ARN that SageMaker should assume:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789:role/SageMakerRole
```

### Training job fails
Check CloudWatch logs:
```bash
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name fraud-detection-xxx/algo-1-xxx
```

### Endpoint returns errors
Verify the endpoint is InService:
```bash
aws sagemaker describe-endpoint --endpoint-name fraud-detector-endpoint
```
