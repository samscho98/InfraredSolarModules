import solar_model as sm

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Panel Anomaly Detection')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'evaluate_checkpoint'],
                      default='evaluate', help='Mode of operation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, label_encoder, history = sm.train_and_evaluate()
    elif args.mode == 'evaluate':
        accuracy, report = sm.evaluate_existing_model()
    else:  # evaluate_checkpoint
        model = sm.load_best_model()
        accuracy, report = sm.evaluate_existing_model(model_path=sm.find_best_checkpoint())