import json
import os
from collections import defaultdict
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import requests
from io import BytesIO
import sentencepiece as spm
import tempfile


class VISTDataset(Dataset):
    """
    ✅ VIST Dataset Loader with SentencePiece Subword Tokenization
    (Word-level vocab completely removed)
    """

    def __init__(
        self,
        dataset_path,
        json_file,
        transform=None,
        img_size=224,
        build_tokenizer=True,
        sp_model=None,
        vocab_size=8000,
        max_len=200
    ):
        self.dataset_path = dataset_path
        self.json_file = json_file
        self.img_size = img_size
        self.max_len = max_len

        # ---------------- IMAGE FOLDER ----------------
        if "train" in json_file:
            self.image_folder = "train_data"
        elif "val" in json_file:
            self.image_folder = "val_data"
        elif "test" in json_file:
            self.image_folder = "test_data"
        else:
            self.image_folder = None

        # ---------------- TRANSFORMS ----------------
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"📂 Loading: {os.path.join(dataset_path, json_file)}")

        with open(os.path.join(dataset_path, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ---------------- PARSE JSON ----------------
        images_list = data.get("images", [])
        annotations_raw = data.get("annotations", [])

        # ---------------- PHOTO → URL ----------------
        self.photo_url_map = {}
        for img in images_list:
            pid = img.get("id") or img.get("photo_flickr_id")
            url = img.get("url_o")
            if pid and url:
                self.photo_url_map[str(pid)] = url

        # ---------------- FLATTEN ANNOTATIONS ----------------
        flat_annotations = []
        for a in annotations_raw:
            if isinstance(a, list):
                flat_annotations.extend(a)
            elif isinstance(a, dict):
                flat_annotations.append(a)

        # ---------------- GROUP STORIES ----------------
        stories_dict = defaultdict(list)
        for ann in flat_annotations:
            stories_dict[ann.get("album_id")].append(ann)

        self.stories = []
        for album_id, anns in stories_dict.items():
            try:
                anns.sort(key=lambda x: int(x.get("worker_arranged_photo_order", 0)))
            except:
                pass

            if len(anns) >= 5:
                self.stories.append({
                    "story_id": album_id,
                    "photo_ids": [str(a.get("photo_flickr_id")) for a in anns[:5]],
                    "texts": [a.get("original_text", "") for a in anns[:5]]
                })

        print(f"✅ Built {len(self.stories)} stories")

        # ---------------- SENTENCEPIECE ----------------
        if sp_model is not None:
            self.sp = sp_model
            print("📚 Using shared SentencePiece tokenizer")
        else:
            self.sp = spm.SentencePieceProcessor()
            if build_tokenizer:
                self._train_sentencepiece(vocab_size)

        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        print(f"🔤 SentencePiece vocab size: {self.sp.get_piece_size()}")

    # ======================================================
    # TRAIN SENTENCEPIECE
    # ======================================================
    def _train_sentencepiece(self, vocab_size):
        print("🔤 Training SentencePiece tokenizer...")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            for s in self.stories:
                f.write(" ".join(s["texts"]) + "\n")
            corpus_path = f.name

        model_prefix = os.path.join(tempfile.gettempdir(), "vist_sp")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )

        self.sp.load(model_prefix + ".model")

    # ======================================================
    # IMAGE LOADER
    # ======================================================
    def _load_image(self, photo_id):
        if self.image_folder:
            local_path = os.path.join(
                self.dataset_path,
                self.image_folder,
                f"{photo_id}.jpg"
            )
            if os.path.exists(local_path):
                try:
                    img = Image.open(local_path).convert("RGB")
                    return self.transform(img)
                except:
                    pass

        if photo_id in self.photo_url_map:
            try:
                r = requests.get(self.photo_url_map[photo_id], timeout=10)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                return self.transform(img)
            except:
                pass

        return torch.zeros(3, self.img_size, self.img_size)

    # ======================================================
    # DATASET API
    # ======================================================
    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        images = torch.stack([
            self._load_image(pid) for pid in story["photo_ids"]
        ])

        text = " ".join(story["texts"])
        token_ids = [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]

        token_ids = token_ids[:self.max_len]
        token_ids += [self.pad_id] * (self.max_len - len(token_ids))

        return {
            "images": images,
            "texts": torch.tensor(token_ids, dtype=torch.long)
        }




# ============================================================================
# USAGE EXAMPLE:
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("✅ VIST Dataset Loader - URL-based Image Loading")
    print("="*70 + "\n")

    # Train dataset (builds vocabulary)
    train_dataset = VISTDataset(
        dataset_path="/teamspace/studios/this_studio/vist_data",
        json_file="train.story-in-sequence.json",
        build_tokenizer=True,
        vocab_size=8000
    )

    print(f"\n📊 TRAIN DATASET:")
    print(f"   Stories: {len(train_dataset)}")
    print(f"   Image URLs: {len(train_dataset.photo_url_map)} photos")

    # Val/Test datasets (reuse train vocabulary)
    print("\n📖 Loading validation dataset...")
    val_dataset = VISTDataset(
        dataset_path="/teamspace/studios/this_studio/vist_data",
        json_file="val.story-in-sequence.json",
        build_tokenizer=False,
        sp_model=train_dataset.sp
    )


    print(f"\n📊 VAL DATASET:")
    print(f"   Stories: {len(val_dataset)}")

    print("\n📖 Loading test dataset...")
    test_dataset = VISTDataset(
        dataset_path="/teamspace/studios/this_studio/vist_data",
        json_file="test.story-in-sequence.json",
        build_tokenizer=False,
        sp_model=train_dataset.sp
    )

    print(f"\n📊 TEST DATASET:")
    print(f"   Stories: {len(test_dataset)}")

    # Test loading a sample
    print("\n🔄 Testing sample loading...")
    try:
        sample = train_dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Images shape: {sample['images'].shape}")
        print(f"   Images min/max: {sample['images'].min():.2f} / {sample['images'].max():.2f}")
        print(f"   Texts shape: {sample['texts'].shape}")
        print(f"\n🎉 DataLoaders ready for training! 🚀")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        import traceback
        traceback.print_exc()

  from tqdm import tqdm

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        images = batch["images"].to(device)
        texts  = batch["texts"].to(device)

        # Teacher forcing
        inputs = texts[:, :-1]
        targets = texts[:, 1:]

        logits = model(images, inputs)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["images"].to(device)
            texts  = batch["texts"].to(device)

            inputs = texts[:, :-1]
            targets = texts[:, 1:]

            logits = model(images, inputs)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)
