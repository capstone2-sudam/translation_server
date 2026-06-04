import torch

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data.dataset import SignLanguageDataset
from data.collate import SignLanguageCollateFn
from models.model import SignLanguageTranslator


def test_real_batch():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "gogamza/kobart-base-v2"
    )

    dataset = SignLanguageDataset(
        json_dir="./dataset/merged_dataset",
        mode="sensor_fusion"
    )


    # ===== 여기 추가 =====
    sample = dataset[0]

    print("\n===== Single Sample Shape =====")
    print("pose :", sample[0].shape)
    print("face :", sample[1].shape)
    print("left :", sample[2].shape)
    print("right :", sample[3].shape)
    print("both :", sample[4].shape)
    print("sensor :", sample[5].shape)
    print("text :", sample[6])
    # =======================

    print("\n===== Feature Dimension Check =====")

    print("pose_dim =", sample[0].shape[-1])
    print("face_dim =", sample[1].shape[-1])
    print("left_dim =", sample[2].shape[-1])
    print("right_dim =", sample[3].shape[-1])
    print("both_dim =", sample[4].shape[-1])
    print("sensor_dim =", sample[5].shape[-1])

    collate_fn = SignLanguageCollateFn(tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    print("\n===== Batch Shapes =====")

    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, v.shape)

    pose_dim = batch["pose_inputs"].shape[-1]
    face_dim = batch["face_inputs"].shape[-1]
    left_dim = batch["left_hand_inputs"].shape[-1]
    right_dim = batch["right_hand_inputs"].shape[-1]
    both_dim = batch["hand_vision_inputs"].shape[-1]
    sensor_dim = batch["sensor_inputs"].shape[-1]

    model = SignLanguageTranslator(
        pose_dim=pose_dim,
        face_dim=face_dim,
        left_dim=left_dim,
        right_dim=right_dim,
        both_dim=both_dim,
        sensor_dim=sensor_dim,
        mode="sensor_fusion",
        encoder_type="gru"
    ).to(device)

    batch = {
        k: v.to(device)
        if torch.is_tensor(v)
        else v
        for k, v in batch.items()
    }

    outputs = model(
        pose_inputs=batch["pose_inputs"],
        face_inputs=batch["face_inputs"],
        left_inputs=batch["left_hand_inputs"],
        right_inputs=batch["right_hand_inputs"],
        both_inputs=batch["hand_vision_inputs"],
        sensor_inputs=batch["sensor_inputs"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )

    print("\n===== Forward Result =====")

    print("Loss:", outputs.loss.item())
    print("Logits Shape:", outputs.logits.shape)

    assert outputs.loss is not None

    print("\nForward Pass Success")


@torch.no_grad()
def test_real_generate():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "gogamza/kobart-base-v2"
    )

    dataset = SignLanguageDataset(
        json_dir="./dataset/merged_dataset",
        mode="sensor_fusion"
    )

    collate_fn = SignLanguageCollateFn(tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    pose_dim = batch["pose_inputs"].shape[-1]
    face_dim = batch["face_inputs"].shape[-1]
    left_dim = batch["left_hand_inputs"].shape[-1]
    right_dim = batch["right_hand_inputs"].shape[-1]
    both_dim = batch["hand_vision_inputs"].shape[-1]
    sensor_dim = batch["sensor_inputs"].shape[-1]

    model = SignLanguageTranslator(
        pose_dim=pose_dim,
        face_dim=face_dim,
        left_dim=left_dim,
        right_dim=right_dim,
        both_dim=both_dim,
        sensor_dim=sensor_dim,
        mode="sensor_fusion",
        encoder_type="gru"
    ).to(device)

    batch = {
        k: v.to(device)
        if torch.is_tensor(v)
        else v
        for k, v in batch.items()
    }

    generated_ids = model.generate(
        pose_inputs=batch["pose_inputs"],
        face_inputs=batch["face_inputs"],
        left_inputs=batch["left_hand_inputs"],
        right_inputs=batch["right_hand_inputs"],
        both_inputs=batch["hand_vision_inputs"],
        sensor_inputs=batch["sensor_inputs"],
        attention_mask=batch["attention_mask"],
        max_length=30
    )

    print("\n===== Generate Result =====")
    print("Generated Shape:", generated_ids.shape)

    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    for i, text in enumerate(decoded):
        print(f"[{i}] {text}")


if __name__ == "__main__":
    test_real_batch()
    test_real_generate()