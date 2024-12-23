import solar_model as sm
import config as cfg

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Panel Anomaly Detection')
    parser.add_argument(
        '--mode', 
        choices=['train', 'evaluate', 'evaluate_checkpoint', 'predict'],
        default='evaluate', 
        help='Mode of operation'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to the image for prediction (required for predict mode)',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Optional: Custom path to the model file'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'predict' and not args.image_path:
        parser.error("--image_path is required when using predict mode")
    
    if args.mode == 'train':
        model, label_encoder, history = sm.train_and_evaluate()
        
    elif args.mode == 'evaluate':
        accuracy, report = sm.evaluate_existing_model()
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)
        
    elif args.mode == 'evaluate_checkpoint':
        model = sm.load_best_model()
        accuracy, report = sm.evaluate_existing_model(
            model_path=sm.find_best_checkpoint()
        )
        print(f"Checkpoint Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)
        
    elif args.mode == 'predict':
        prediction = sm.predict_single_image(
            image_path=args.image_path,
            model_path=args.model_path if args.model_path else cfg.MODEL_SAVE_PATH
        )
        print(f"\nPrediction for image {args.image_path}:")
        print(f"Predicted class: {prediction}")