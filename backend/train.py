
from pathlib import Path
import io
import random
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from models.unet import UNet


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "external" / "KuroSiwo" / "train_GRD"

MODEL_DIR = ROOT / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODEL_DIR / "flood_unet.pth"
SPLIT_FILE = MODEL_DIR / "kuro_siwo_split.pt"

SEED = 42

# Full indexed dataset. No temporary sample limit.
EPOCHS = 5

BATCH_SIZE = 2
LEARNING_RATE = 1e-3

NUM_WORKERS = 0

DEVICE = torch.device("cpu")

INPUT_CHANNELS = 7
OUTPUT_CHANNELS = 1

SAR_NAMES = [
    "flood_vv.npy",
    "flood_vh.npy",
    "sec1_vv.npy",
    "sec1_vh.npy",
    "sec2_vv.npy",
    "sec2_vh.npy",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class KuroSiwoDataset(Dataset):

    def __init__(self, archive_dir):

        self.archive_dir = Path(archive_dir)

        self.archives = sorted(
            self.archive_dir.glob("*.tar")
        )

        if not self.archives:
            raise RuntimeError(
                f"No TAR archives found in: {self.archive_dir}"
            )

        # Each entry:
        # (archive_path, sample_id, member_lookup)
        self.samples = []

        print("\nIndexing Kuro Siwo archives...")

        for archive_path in self.archives:

            print(f"Indexing: {archive_path.name}")

            with tarfile.open(archive_path, "r") as tar:

                member_lookup = {}

                for member in tar:

                    if not member.isfile():
                        continue

                    basename = Path(member.name).name

                    member_lookup[basename] = member.name

                sample_ids = set()

                for basename in member_lookup:

                    if basename.endswith(".info.json"):

                        sample_id = basename.removesuffix(
                            ".info.json"
                        )

                        sample_ids.add(sample_id)

                for sample_id in sorted(sample_ids):

                    self.samples.append(
                        (
                            archive_path,
                            sample_id,
                            member_lookup
                        )
                    )

        if not self.samples:

            raise RuntimeError(
                "No Kuro Siwo samples were indexed."
            )

        print(f"\nArchives found: {len(self.archives)}")
        print(f"Samples indexed: {len(self.samples)}")

    def __len__(self):

        return len(self.samples)

    # --------------------------------------------------------
    # ARRAY SHAPE HANDLING
    # --------------------------------------------------------

    @staticmethod
    def to_2d(array, name):

        array = np.asarray(array)

        # Remove singleton dimensions.
        array = np.squeeze(array)

        if array.ndim != 2:

            raise ValueError(
                f"{name} must be 2D after squeezing. "
                f"Received shape: {array.shape}"
            )

        return array.astype(np.float32)

    # --------------------------------------------------------
    # LOAD ARRAY FROM TAR
    # --------------------------------------------------------

    @staticmethod
    def load_array(
        tar,
        sample_id,
        suffix,
        member_lookup
    ):

        member_name = f"{sample_id}.{suffix}"

        # Use pre-indexed member path.
        actual_name = member_lookup.get(member_name)

        if actual_name is None:

            raise RuntimeError(
                f"Missing archive member: {member_name}"
            )

        try:

            file_obj = tar.extractfile(actual_name)

            if file_obj is None:

                raise RuntimeError(
                    f"Could not read archive member: "
                    f"{actual_name}"
                )

            array_bytes = file_obj.read()

            array = np.load(
                io.BytesIO(array_bytes),
                allow_pickle=False
            )

            return np.asarray(
                array,
                dtype=np.float32
            )

        except Exception as exc:

            raise RuntimeError(
                f"Failed loading {member_name}: {exc}"
            ) from exc

    # --------------------------------------------------------
    # SAR NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def normalize_sar(array):

        array = array.astype(np.float32)

        valid = np.isfinite(array)

        if not valid.any():

            return np.zeros_like(
                array,
                dtype=np.float32
            )

        low, high = np.percentile(
            array[valid],
            [2, 98]
        )

        array = np.nan_to_num(
            array,
            nan=float(low),
            posinf=float(high),
            neginf=float(low)
        )

        if high > low:

            array = np.clip(
                (array - low) / (high - low),
                0,
                1
            )

        else:

            array[:] = 0

        return array.astype(np.float32)

    # --------------------------------------------------------
    # DEM NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def normalize_dem(dem):

        dem = np.nan_to_num(
            dem,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        mean = float(dem.mean())
        std = float(dem.std())

        if std > 1e-6:

            dem = (dem - mean) / std

        else:

            dem[:] = 0

        return dem.astype(np.float32)

    # --------------------------------------------------------
    # LOAD ONE SAMPLE
    # --------------------------------------------------------

    def __getitem__(self, index):

        archive_path, sample_id, member_lookup = (
            self.samples[index]
        )

        with tarfile.open(
            archive_path,
            "r"
        ) as tar:

            # -----------------------------------------------
            # Six SAR input channels
            # -----------------------------------------------

            sar_channels = []

            for suffix in SAR_NAMES:

                band = self.load_array(
                    tar,
                    sample_id,
                    suffix,
                    member_lookup
                )

                band = self.to_2d(
                    band,
                    suffix
                )

                band = self.normalize_sar(
                    band
                )

                sar_channels.append(band)

            # -----------------------------------------------
            # DEM: seventh input channel
            # -----------------------------------------------

            dem = self.load_array(
                tar,
                sample_id,
                "dem.npy",
                member_lookup
            )

            dem = self.to_2d(
                dem,
                "dem"
            )

            dem = self.normalize_dem(
                dem
            )

            # -----------------------------------------------
            # Ensure all input channels have same dimensions
            # -----------------------------------------------

            reference_shape = sar_channels[0].shape

            for channel_index, channel in enumerate(
                sar_channels
            ):

                if channel.shape != reference_shape:

                    raise ValueError(
                        f"SAR channel shape mismatch in "
                        f"{sample_id}: "
                        f"{SAR_NAMES[channel_index]} "
                        f"has shape {channel.shape}, "
                        f"expected {reference_shape}"
                    )

            if dem.shape != reference_shape:

                raise ValueError(
                    f"DEM shape mismatch in {sample_id}: "
                    f"{dem.shape}, expected {reference_shape}"
                )

            # -----------------------------------------------
            # Stack image
            # Shape: [7, H, W]
            # -----------------------------------------------

            image = np.stack(
                sar_channels + [dem],
                axis=0
            ).astype(np.float32)

            # -----------------------------------------------
            # Load flood mask
            # -----------------------------------------------

            mask = self.load_array(
                tar,
                sample_id,
                "mask.npy",
                member_lookup
            )

            mask = self.to_2d(
                mask,
                "mask"
            )

            # -----------------------------------------------
            # Load valid mask
            # -----------------------------------------------

            valid_mask = self.load_array(
                tar,
                sample_id,
                "valid_mask.npy",
                member_lookup
            )

            valid_mask = self.to_2d(
                valid_mask,
                "valid_mask"
            )

        # ----------------------------------------------------
        # Validate target dimensions
        # ----------------------------------------------------

        if mask.shape != reference_shape:

            raise ValueError(
                f"Flood mask shape mismatch for {sample_id}: "
                f"{mask.shape}, expected {reference_shape}"
            )

        if valid_mask.shape != reference_shape:

            raise ValueError(
                f"Valid mask shape mismatch for {sample_id}: "
                f"{valid_mask.shape}, expected {reference_shape}"
            )

        # ----------------------------------------------------
        # Binary flood segmentation
        #
        # 0 = no water
        # 1 = permanent water
        # 2 = flood
        #
        # Target:
        # flood = 1
        # other classes = 0
        # ----------------------------------------------------

        target = (
            mask == 2
        ).astype(np.float32)

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------

        valid = (
            (valid_mask == 1)
            & np.isfinite(mask)
            & np.isin(mask, [0, 1, 2])
        )

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        target = target.astype(np.float32)

        valid = valid.astype(np.float32)

        # ----------------------------------------------------
        # Final expected shapes:
        #
        # image:  [7, H, W]
        # target: [1, H, W]
        # valid:  [1, H, W]
        # ----------------------------------------------------

        if image.ndim != 3:

            raise ValueError(
                f"Image must be 3D, got {image.shape}"
            )

        if target.ndim != 2:

            raise ValueError(
                f"Target must be 2D, got {target.shape}"
            )

        if valid.ndim != 2:

            raise ValueError(
                f"Valid mask must be 2D, got {valid.shape}"
            )

        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(target[None].copy()),
            torch.from_numpy(valid[None].copy())
        )


# ============================================================
# MASKED LOSS
# ============================================================

def calculate_masked_loss(
    logits,
    targets,
    valid,
    criterion
):

    # Expected:
    # logits:  [B, 1, H, W]
    # targets: [B, 1, H, W]
    # valid:   [B, 1, H, W]

    if logits.shape != targets.shape:

        raise ValueError(
            f"Logits shape {logits.shape} "
            f"does not match target shape {targets.shape}"
        )

    if targets.shape != valid.shape:

        raise ValueError(
            f"Target shape {targets.shape} "
            f"does not match valid mask shape {valid.shape}"
        )

    pixel_loss = criterion(
        logits,
        targets
    )

    valid_pixels = valid.sum()

    # Skip batches with no valid pixels.
    if valid_pixels.item() == 0:

        return None

    loss = (
        (pixel_loss * valid).sum()
        / valid_pixels
    )

    return loss


# ============================================================
# TRAINING
# ============================================================

def main():

    start_time = time.time()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KuroSiwoDataset(
        DATA_DIR
    )

    if len(dataset) < 10:

        raise RuntimeError(
            "Not enough training samples were found."
        )

    # --------------------------------------------------------
    # Reproducible 80/20 split
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_size = int(
        len(dataset) * 0.8
    )

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # --------------------------------------------------------
    # Save split
    # --------------------------------------------------------

    torch.save(
        {
            "train_indices": train_dataset.indices,
            "val_indices": val_dataset.indices,
            "seed": SEED,
            "dataset": "Kuro Siwo GRD",
            "total_samples": len(dataset)
        },
        SPLIT_FILE
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n======================================")
    print("       FLOODSENSE MODEL TRAINING")
    print("======================================")

    print(f"Device: {DEVICE}")
    print(f"Input channels: {INPUT_CHANNELS}")
    print(f"Total indexed samples: {len(dataset)}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    print("======================================\n")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_loss_total = 0.0
        train_batches = 0
        skipped_train_batches = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} "
            f"- Training"
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1
        ):

            images, targets, valid = batch

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            valid = valid.to(DEVICE)

            # -----------------------------------------------
            # Shape checks
            # -----------------------------------------------

            if images.ndim != 4:

                raise ValueError(
                    f"Model input must be 4D "
                    f"[B,C,H,W], got {images.shape}"
                )

            if images.shape[1] != INPUT_CHANNELS:

                raise ValueError(
                    f"Expected {INPUT_CHANNELS} input channels, "
                    f"got {images.shape[1]}"
                )

            if targets.ndim != 4:

                raise ValueError(
                    f"Target must be 4D [B,1,H,W], "
                    f"got {targets.shape}"
                )

            if valid.ndim != 4:

                raise ValueError(
                    f"Valid mask must be 4D [B,1,H,W], "
                    f"got {valid.shape}"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            # -----------------------------------------------
            # Forward pass
            # -----------------------------------------------

            logits = model(images)

            # -----------------------------------------------
            # Loss
            # -----------------------------------------------

            loss = calculate_masked_loss(
                logits,
                targets,
                valid,
                criterion
            )

            if loss is None:

                skipped_train_batches += 1
                continue

            # -----------------------------------------------
            # Backpropagation
            # -----------------------------------------------

            loss.backward()

            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

            # -----------------------------------------------
            # Progress
            # -----------------------------------------------

            if batch_index % 100 == 0:

                print(
                    f"Training batch "
                    f"{batch_index}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_train_loss = (
            train_loss_total / max(train_batches, 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0.0
        val_batches = 0
        skipped_val_batches = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} "
            f"- Validation"
        )

        with torch.no_grad():

            for batch_index, batch in enumerate(
                val_loader,
                start=1
            ):

                images, targets, valid = batch

                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                valid = valid.to(DEVICE)

                logits = model(images)

                loss = calculate_masked_loss(
                    logits,
                    targets,
                    valid,
                    criterion
                )

                if loss is None:

                    skipped_val_batches += 1
                    continue

                val_loss_total += loss.item()
                val_batches += 1

                if batch_index % 100 == 0:

                    print(
                        f"Validation batch "
                        f"{batch_index}/{len(val_loader)} "
                        f"| Loss: {loss.item():.4f}"
                    )

        average_val_loss = (
            val_loss_total / max(val_batches, 1)
        )

        epoch_duration = (
            time.time() - epoch_start
        )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print("\n--------------------------------------")

        print(
            f"Epoch {epoch + 1}/{EPOCHS} completed"
        )

        print(
            f"Training loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{average_val_loss:.4f}"
        )

        print(
            f"Training batches processed: "
            f"{train_batches}"
        )

        print(
            f"Training batches skipped: "
            f"{skipped_train_batches}"
        )

        print(
            f"Validation batches processed: "
            f"{val_batches}"
        )

        print(
            f"Validation batches skipped: "
            f"{skipped_val_batches}"
        )

        print(
            f"Epoch duration: "
            f"{epoch_duration / 60:.2f} minutes"
        )

        print("--------------------------------------")

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            val_batches > 0
            and average_val_loss < best_val_loss
        ):

            best_val_loss = average_val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": INPUT_CHANNELS,
                    "out_channels": OUTPUT_CHANNELS,
                    "epochs_completed": epoch + 1,
                    "training_samples": train_size,
                    "validation_samples": val_size,
                    "seed": SEED,
                    "dataset": "Kuro Siwo GRD",
                    "best_validation_loss": best_val_loss
                },
                CHECKPOINT
            )

            print(
                "\nBest model checkpoint saved:"
            )

            print(CHECKPOINT)

        # ----------------------------------------------------
        # No valid validation batches
        # ----------------------------------------------------

        if val_batches == 0:

            raise RuntimeError(
                "No valid validation batches were processed. "
                "Training stopped because validation loss "
                "could not be calculated."
            )

    # ========================================================
    # FINISHED
    # ========================================================

    total_duration = (
        time.time() - start_time
    )

    print("\n======================================")
    print("         TRAINING FINISHED")
    print("======================================")

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Model saved at: {CHECKPOINT}"
    )

    print(
        f"Total training duration: "
        f"{total_duration / 3600:.2f} hours"
    )

    print("======================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
