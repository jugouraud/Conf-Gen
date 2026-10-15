import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.embeddings.coco import (
    _upsert_annotations,
    build_parser,
    create_empty_coco_database,
    fill_coco_database,
    fill_coco_database_on_gpu,
)


class CocoDatabaseTests(unittest.TestCase):
    def test_empty_database_uses_image_id_as_the_primary_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = create_empty_coco_database(Path(directory) / "coco.sqlite3")
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM captions").fetchone()[0], 0)
                columns = connection.execute("PRAGMA table_info(images)").fetchall()
            self.assertEqual({column[1] for column in columns if column[5]}, {"image_id"})

    def test_unchanged_annotations_do_not_rewrite_existing_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = create_empty_coco_database(Path(directory) / "coco.sqlite3")
            annotations = {
                "images": [{"id": 7, "file_name": "7.jpg"}],
                "captions": [{"id": 101, "image_id": 7, "caption": "A caption."}],
            }
            with closing(sqlite3.connect(database)) as connection:
                _upsert_annotations(connection, annotations)
                connection.execute("UPDATE captions SET text_context = ? WHERE caption_id = 101", (b"keep",))
                connection.commit()
                changes_before = connection.total_changes
                _upsert_annotations(connection, annotations)
                self.assertEqual(connection.total_changes, changes_before)
                self.assertEqual(
                    connection.execute("SELECT text_context FROM captions WHERE caption_id = 101").fetchone()[0],
                    b"keep",
                )
                changed = {**annotations, "captions": [
                    {"id": 101, "image_id": 7, "caption": "Changed caption."}
                ]}
                _upsert_annotations(connection, changed)
                self.assertEqual(connection.total_changes, changes_before + 1)

    def test_fill_links_multiple_captions_to_one_image_and_stores_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "captions.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 7, "file_name": "000000000007.jpg"}],
                        "annotations": [
                            {"id": 101, "image_id": 7, "caption": "First caption."},
                            {"id": 102, "image_id": 7, "caption": "Second caption."},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            image = root / "images" / "000000000007.jpg"
            image.parent.mkdir()
            image.touch()
            vector = np.ones((77, 768), dtype=np.float32)

            def extract(caption, image_path, output_directory, **_):
                output_directory.mkdir(parents=True)
                output = output_directory / "text_context.npy"
                np.save(output, vector)
                return {"text_context": output}

            database = root / "coco.sqlite3"
            with patch("backend.embeddings.coco.extract_inner_embeddings", side_effect=extract):
                self.assertEqual(
                    fill_coco_database(
                        annotations_path=annotations, images_directory=image.parent, database_path=database, device="cuda"
                    ),
                    2,
                )

            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT image_id, caption, text_context_shape, embedding_model FROM captions ORDER BY caption_id"
                ).fetchall()
            self.assertEqual([row[0] for row in rows], [7, 7])
            self.assertEqual([row[1] for row in rows], ["First caption.", "Second caption."])
            self.assertEqual([row[2] for row in rows], ["[77, 768]", "[77, 768]"])
            self.assertTrue(all(row[3] for row in rows))

    def test_gpu_job_processes_only_ten_selected_captions_and_merges_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for image_id in (1, 2):
                (images / f"{image_id}.jpg").touch()
            annotations = root / "captions.json"
            annotations.write_text(json.dumps({
                "images": [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in (1, 2)],
                "annotations": [
                    {"id": caption_id, "image_id": 1 if caption_id < 7 else 2,
                     "caption": f"COCO caption {caption_id}"}
                    for caption_id in range(1, 12)
                ],
            }), encoding="utf-8")
            database = root / "local.sqlite3"
            remote_root = root / "remote"
            remote_root.mkdir()
            vector = np.ones((77, 768), dtype=np.float32)
            command_names = []
            selected_batches = []

            def extract(caption, image_path, output_directory, **_):
                output_directory.mkdir(parents=True)
                output = output_directory / "text_context.npy"
                np.save(output, vector)
                return {"text_context": output}

            def run_command(command, **options):
                command_names.append(command[3])
                if command[3] == "exec":
                    self.assertEqual(options["timeout"], 660)
                if command[3] == "upload":
                    shutil.rmtree(remote_root)
                    remote_root.mkdir()
                    with zipfile.ZipFile(command[-2]) as archive:
                        archive.extractall(remote_root)
                        selected = json.loads(archive.read("captions.json"))
                        selected_batches.append({row["id"] for row in selected["annotations"]})
                        expected_image = f"images/{selected['annotations'][0]['image_id']}.jpg"
                        self.assertEqual(
                            {name for name in archive.namelist() if name.startswith("images/")},
                            {expected_image},
                        )
                elif command[3] == "exec":
                    if len(selected_batches) == 2:
                        raise subprocess.CalledProcessError(1, command)
                    with patch("backend.embeddings.coco.extract_inner_embeddings", side_effect=extract):
                        fill_coco_database(
                            annotations_path=remote_root / "captions.json",
                            images_directory=remote_root / "images",
                            database_path=remote_root / "coco.sqlite3",
                            device="cuda",
                        )
                elif command[3] == "download":
                    shutil.copy2(remote_root / "coco.sqlite3", command[-1])

            with patch("backend.embeddings.coco.os", SimpleNamespace(name="posix")), patch(
                "backend.gpu.client._find_colab_command", return_value="colab"
            ), patch("backend.gpu.client._start_session") as start_session, patch(
                "backend.gpu.client._install_remote_dependencies"
            ), patch("backend.gpu.client._stop_session") as stop_session, patch(
                "backend.gpu.client._run_command", side_effect=run_command
            ), patch("backend.embeddings.coco.time.sleep"):
                count = fill_coco_database_on_gpu(
                    annotations_path=annotations, images_directory=images,
                    database_path=database, limit=10, batch_size=6,
                )
            self.assertEqual(count, 10)
            self.assertEqual(selected_batches, [
                set(range(1, 7)), set(range(7, 11)), set(range(7, 11)),
            ])
            self.assertEqual(command_names, [
                "upload", "exec", "download", "upload", "exec", "upload", "exec", "download",
            ])
            self.assertEqual(start_session.call_count, 2)
            self.assertEqual(stop_session.call_count, 2)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*), COUNT(text_context) FROM captions").fetchone(),
                    (11, 10),
                )
                self.assertIsNone(connection.execute(
                    "SELECT text_context FROM captions WHERE caption_id = 11"
                ).fetchone()[0])

    def test_worker_reuses_downloaded_model_between_batches(self) -> None:
        from backend.gpu import coco_worker as colab_coco_worker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            (work / "models").mkdir(parents=True)
            (work / "models" / "cached.bin").write_bytes(b"model")
            (work / "stale.txt").write_text("old")
            archive_path = root / "job.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("manifest.json", json.dumps({
                    "annotations": "captions.json", "images_directory": "images",
                    "model_id": "example/model",
                }))
            torch = ModuleType("torch")
            torch.cuda = SimpleNamespace(is_available=lambda: True)

            def fill(**options):
                (work / "coco.sqlite3").write_bytes(b"result")

            with patch.dict(sys.modules, {"torch": torch}), patch.object(
                colab_coco_worker, "ARCHIVE_PATH", archive_path
            ), patch.object(colab_coco_worker, "WORK_DIRECTORY", work), patch.object(
                colab_coco_worker, "OUTPUT_DATABASE_PATH", root / "output.sqlite3"
            ), patch("backend.embeddings.coco.fill_coco_database", side_effect=fill):
                colab_coco_worker.main()

            self.assertTrue((work / "models" / "cached.bin").is_file())
            self.assertFalse((work / "stale.txt").exists())
            self.assertEqual((root / "output.sqlite3").read_bytes(), b"result")

    def test_gpu_aliases_parse(self) -> None:
        parser = build_parser()
        for flag in ("--gpu", "-gpu"):
            self.assertTrue(parser.parse_args([flag]).gpu)

    def test_gpu_explains_the_windows_wsl_requirement(self) -> None:
        with patch("backend.embeddings.coco.os.name", "nt"), self.assertRaisesRegex(RuntimeError, "POSIX termios"):
            fill_coco_database_on_gpu()


if __name__ == "__main__":
    unittest.main()
