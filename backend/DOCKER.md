# Docker usage

## Start the application

```bash
docker compose up --build app
```

API will be at http://localhost:8000. Dummy weights are created on first run if missing.

## Train on MURA dataset

1. **Set MURA path** – create `.env` in this directory (or copy from `.env.example`):

   ```
   MURA_DATA_DIR=C:/Users/abhij/Downloads/mura_dataset/MURA-v1.1
   ```

   Use forward slashes on Windows. Or place MURA-v1.1 in `./mura_dataset` and omit `MURA_DATA_DIR`.

2. **Run training**:

   ```bash
   docker compose --profile train run --rm train
   ```

   This runs `train_body_part.py` on the MURA dataset and writes `weights/body_part_dinov2_head.pt` into the shared `weights` volume.

3. **Restart the app** to load the new head:

   ```bash
   docker compose restart app
   ```

## Full flow (build, train, run)

```bash
# 1. Create .env with MURA_DATA_DIR pointing to your MURA-v1.1 folder
# 2. Build and start app
docker compose up -d --build app

# 3. Train (downloads DINOv2 and MURA on first run)
docker compose --profile train run --rm train

# 4. Restart app to use trained weights
docker compose restart app
```
