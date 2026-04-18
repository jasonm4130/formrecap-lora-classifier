"""Download a trained LoRA adapter from the Modal volume to local disk."""

import click
import modal


@click.command()
@click.option("--run-id", default="baseline-3b")
@click.option("--local-dir", default="adapter")
def main(run_id: str, local_dir: str):
    vol = modal.Volume.from_name("formrecap-lora")
    remote_path = f"runs/{run_id}/adapter"

    # List files in the adapter directory
    entries = list(vol.listdir(remote_path))
    print(f"Found {len(entries)} files in volume:{remote_path}/")

    from pathlib import Path

    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        if entry.type == modal.volume.FileEntryType.FILE:
            filename = entry.path.split("/")[-1]
            remote_file = f"{remote_path}/{filename}"
            local_file = out / filename
            print(f"  Downloading {entry.path} ({entry.size} bytes)")
            with open(local_file, "wb") as f:
                for chunk in vol.read_file(remote_file):
                    f.write(chunk)

    print(f"Done. Adapter saved to {out}/")


if __name__ == "__main__":
    main()
