import argparse

def get_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="CQT-CycleGAN", help="model name or version")
    parser.add_argument("--num_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--output_channels", type=int, default=3, help="number of output channels")
    parser.add_argument("--norm_type", type=str, default="instancenorm", help="type of normalization")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lr_sgd", type=float, default=0.0001, help="SGD: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--checkpoints_keep", type=int, default=3, help="maximum checkpoints to keep")
    parser.add_argument("--checkpoint_epochs", type=int, default=1, help="number of epochs to leave between checkpoints")
    parser.add_argument("--lambd", type=int, default=10, help="Lambda for calculating loss")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lambda_identity", type=int, default=1, help="Lambda Identity")
    parser.add_argument("--lambda_cycle", type=int, default=10, help="Lambda Cycle")
    parser.add_argument("--img_size", type=int, default=352, help="Size of preprocessed image - % 32 == 0")
    parser.add_argument("--img_width", type=int, default=336, help="Width of Image")
    parser.add_argument("--img_height", type=int, default=250, help="Height of Image")
    parser.add_argument("--train_size", type=int, default=75, help="Number of train images")
    parser.add_argument("--sample_data_path", type=str, default="../../dataset/sample/", help="path to root sample data directory")
    parser.add_argument("--input_data_train_path", type=str, default="../../dataset/processedData/trainSet/", help="path to train data")
    parser.add_argument("--input_data_val_path", type=str, default="../../dataset/processedData/testSet/", help="path to val data")
    parser.add_argument("--input_train_piano_path", type=str, default="../../dataset/processedData/trainSet/flute/cqtChunks/", help="path to piano train directory")
    parser.add_argument("--input_train_flute_path", type=str, default="../../dataset/processedData/trainSet/piano/cqtChunks/", help="path to flute train directory")
    parser.add_argument("--input_test_flute_path", type=str, default="../../dataset/processedData/trainSet/piano/cqtChunks/", help="path to flute test directory")
    parser.add_argument("--input_test_piano_path", type=str, default="../../dataset/processedData/trainSet/flute/cqtChunks/", help="path to piano test directory")
    parser.add_argument("--output_path", type=str, default="../../dataset/saved_images/", help="path to directory for storing model output")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader") 
    
    #parser.add_argument("--data_path", type=str, default="data/2-class", help="path to root data directory")
      
    #parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    #parser.add_argument("--sample_interval", type=int, default=400, help="number of batches between image sampling")
    #parser.add_argument("--print_every", type=int, default=50, help="number of iterations between printing training stats")   
    #parser.add_argument("--num_classes", type=int, default=2, help="number of classes for dataset")  
    #parser.add_argument("--eval_mode", type=str, default="val", help="eval mode: val, test, nn")

    return parser.parse_args()