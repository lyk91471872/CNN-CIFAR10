import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
import click

import config as conf
from dataset import create_dataset
from utils.pipeline import Pipeline
from utils.session import SessionTracker

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Train or cross-validate a model on CIFAR-10."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command('grid-search', help='Run grid search for best data augmentation')
@click.argument('epochs', type=int, default=10)
def grid_search(epochs):
    """Run grid search for optimal data augmentation combinations.
    
    EPOCHS: Number of epochs to train each combination (default: 10)
    """
    print(f"\nRunning grid search for optimal data augmentation with {epochs} epochs")
    try:
        from scripts.grid_search_augmentation import main as grid_search_main
        grid_search_main(epochs=epochs)
    except Exception as e:
        print(f"Error during grid search: {e}")
        import traceback
        traceback.print_exc()

@cli.command('search-channels', help='Find optimal channel size for CustomResNet18X')
def search_channel_size():
    """Find the optimal channel size for CustomResNet18X with <5M parameters."""
    print("\nSearching for optimal channel size...")
    try:
        from scripts.search_channel_size import main as search_main
        optimal_x = search_main()
        print(f"\nSearch complete. Optimal channel size: x = {optimal_x}")
        print(f"To use this model, update get_model() in config.py to return CustomResNet18X")
    except Exception as e:
        print(f"Error during channel search: {e}")

@cli.command('train', help='Train the model on full dataset')
def train():
    """Train the model on full dataset."""
    print("\nTraining on full dataset...")
    dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training')
    model = conf.get_model()()  # Instantiate the model
    pipeline = Pipeline(model)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **conf.DATALOADER)
    val_loader = DataLoader(val_dataset, shuffle=False, **conf.DATALOADER)

    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

@cli.command('crossval', help='Run cross-validation')
def crossval():
    """Run cross-validation."""
    print("\nStarting cross-validation...")
    dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training')
    model = conf.get_model()()  # Instantiate the model
    pipeline = Pipeline(model)

    pipeline.cross_validate(dataset)

@cli.command('pdf', help='Generate PDF of test images')
def generate_pdf():
    """Generate PDF of test images."""
    print("\nGenerating PDF of test images...")
    try:
        from scripts.testset2pdf import testset_to_pdf
        
        output_pdf_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "test_images_raw.pdf")
        test_dataset = create_dataset(data_source=conf.TEST_DATA_PATH, mode='test', raw=True)
        testset_to_pdf(test_dataset, output_pdf_path, use_grayscale=False)
        
        print(f"PDF generated successfully: {output_pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

@cli.command('train-pdf', help='Generate PDF of training images (batch 1)')
def generate_train_pdf():
    """Generate PDF of training images (batch 1)."""
    print("\nGenerating PDF of training images (batch 1)...")
    try:
        from scripts.trainingset2pdf import CIFAR10TrainingDatasetRaw, trainingset_to_pdf
        
        training_batch_path = os.path.join(conf.DATA_DIR, 'data_batch_1')
        output_pdf_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "training_images.pdf")
        
        training_dataset = CIFAR10TrainingDatasetRaw(training_batch_path)
        trainingset_to_pdf(training_dataset, output_pdf_path, use_grayscale=False)
        
        print(f"PDF generated successfully: {output_pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

@cli.command('benchmark', help='Run dataloader benchmark')
def benchmark():
    """Run dataloader benchmark."""
    print("\nRunning dataloader benchmark...")
    try:
        from scripts.dataloader_benchmark import main as benchmark_main
        benchmark_main()
    except Exception as e:
        print(f"Error running benchmark: {e}")

@cli.command('normalize', help='Update normalization values')
def normalize():
    """Update normalization values."""
    print("\nUpdating normalization values...")
    try:
        from scripts.update_normalization_values import main as normalize_main
        normalize_main()
    except Exception as e:
        print(f"Error updating normalization values: {e}")

@cli.command('list-sessions', help='List recent training/cross-validation sessions')
@click.option('--model', help='Filter sessions by model name')
@click.option('--type', type=click.Choice(['training', 'crossval']), help='Filter sessions by type')
@click.option('--limit', type=int, default=10, help='Maximum number of sessions to list')
def list_sessions(model, type, limit):
    """List recent training/cross-validation sessions."""
    print("\nListing recent training/cross-validation sessions:")
    sessions = SessionTracker.list_sessions(
        model_name=model,
        session_type=type,
        limit=limit
    )
    
    if not sessions:
        print("No sessions found matching your criteria.")
        return
    
    print(f"\nFound {len(sessions)} session(s):")
    for i, session in enumerate(sessions, 1):
        data = session.data
        print(f"\n{i}. {data['model_name']} - {data['session_type']} - {data['timestamp']}")
        
        if 'metrics' in data:
            metrics = data['metrics']
            if 'best_val_acc' in metrics:
                print(f"   Accuracy: {metrics['best_val_acc']*100:.2f}%")
            if 'epochs' in metrics:
                print(f"   Epochs: {metrics['epochs']}")
            if 'avg_val_acc' in metrics:
                print(f"   Avg CV Accuracy: {metrics['avg_val_acc']*100:.2f}%")
        
        if 'files' in data:
            files = data['files']
            print(f"   Files:")
            for file_type, file_path in files.items():
                print(f"     - {file_type}: {file_path}")

# Add all command aliases in one place
cli.add_command(grid_search, name='g')
cli.add_command(search_channel_size, name='s')
cli.add_command(train, name='t')
cli.add_command(crossval, name='c')
cli.add_command(generate_pdf, name='p')
cli.add_command(generate_train_pdf, name='tp')
cli.add_command(benchmark, name='b')
cli.add_command(normalize, name='n')
cli.add_command(list_sessions, name='l')

def main():
    """Main function to run the CLI."""
    cli()

if __name__ == "__main__":
    main()
