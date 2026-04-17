"""One-shot uploader: push local JSONL datasets to the Modal volume."""

import click
import modal


@click.command()
@click.option("--local-dir", default="data/synthetic")
def main(local_dir: str):
    vol = modal.Volume.from_name("formrecap-lora", create_if_missing=True)
    with vol.batch_upload() as batch:
        batch.put_directory(local_dir, "data")
    print(f"Uploaded {local_dir} -> volume:data/")


if __name__ == "__main__":
    main()
