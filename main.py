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

@cli.command(name='train', help='Train the model on full dataset')
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

@cli.command(name='crossval', help='Run cross-validation')
def crossval():
    """Run cross-validation."""
    print("\nStarting cross-validation...")
    dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training')
    model = conf.get_model()()  # Instantiate the model
    pipeline = Pipeline(model)

    pipeline.cross_validate(dataset)

@cli.command(name='pdf', help='Generate PDF of test images')
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

@cli.command(name='train-pdf', help='Generate PDF of training images (batch 1)')
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

@cli.command(name='benchmark', help='Run dataloader benchmark')
def benchmark():
    """Run dataloader benchmark."""
    print("\nRunning dataloader benchmark...")
    try:
        from scripts.dataloader_benchmark import main as benchmark_main
        benchmark_main()
    except Exception as e:
        print(f"Error running benchmark: {e}")

@cli.command(name='normalize', help='Update normalization values')
def normalize():
    """Update normalization values."""
    print("\nUpdating normalization values...")
    try:
        from scripts.update_normalization_values import main as normalize_main
        normalize_main()
    except Exception as e:
        print(f"Error updating normalization values: {e}")

@cli.command(name='list-sessions', help='List recent training/cross-validation sessions')
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
            print("   Files:")
            for file_type, file_path in data['files'].items():
                print(f"     - {file_type}: {file_path}")

def main():
    # Manual alias mapping for dash-style aliases.
    alias_mapping = {
        '-t': 'train',
        '-c': 'crossval',
        '-p': 'pdf',
        '-tp': 'train-pdf',
        '-b': 'benchmark',
        '-n': 'normalize',
        '-l': 'list-sessions'
    }
    # If the first argument is a dash alias, replace it with its full command name.
    if len(sys.argv) > 1 and sys.argv[1] in alias_mapping:
        sys.argv[1] = alias_mapping[sys.argv[1]]
    cli()

if __name__ == "__main__":
    main()
