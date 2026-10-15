import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from torchvision import datasets, transforms
from torch.utils.data import Subset

from transformers import AutoTokenizer, AutoModelForCausalLM

import time


from transformers import CLIPVisionModel


# 参数
# batch = 2

class I_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        for p in self.vision_model.parameters():
            p.requires_grad = False

        for p in self.vision_model.encoder.layers[-1].parameters():
            p.requires_grad = True


        # self.patch_embed = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=256,
        #     kernel_size=16,
        #     stride=16
        # )
        # self.pos_embed = nn.Parameter(
        #     torch.randn(1,196,256)
        # )
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=256,
        #     nhead=8,
        #     batch_first=True
        # )
        #
        # self.transformer = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=2
        # )
        # self.mm_projector = nn.Linear(256, 896)

    def forward(self,images):
        # images = self.patch_embed(images)
        # PyTorch 2.5.1 的 MPS 卷积反向传播需要连续的输出梯度。
        if images.device.type == "mps" and images.requires_grad:
            images.register_hook(lambda grad: grad.contiguous())
        # images = images.flatten(2)
        # images = images.transpose(1, 2)
        # images = images + self.pos_embed
        # images = self.transformer(images)

        vision_output = self.vision_model(
            pixel_values=images
        )
        image_tokens = vision_output.last_hidden_state
        image_tokens = image_tokens[:, 1:, :]
        # [B, 49, 768]


        return image_tokens


class MiniLLaVA(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.iencoder = I_Encoder()
        self.mm_projector = nn.Linear(
            768,
            self.llm.config.hidden_size
        )

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image>"]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def forward(self,images, questions, answers):

        image_tokens = self.iencoder.forward(images)
        image_embeds = self.mm_projector(image_tokens)

        messages = []
        for question, answer in zip(questions, answers):
            message = [
                {"role": "user", "content": question, },
                {"role": "assistant", "content": answer, },
            ]
            messages.append(message)

        texts = []
        for message in messages:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False
            )
            texts.append(text)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        input_ids = inputs["input_ids"]

        assistant_ids = self.tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False
        )["input_ids"]
        assistant_ids = torch.tensor(
            assistant_ids,
            device=input_ids.device
        )

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_positions = []
        for i in range(input_ids.shape[0]):
            pos = (input_ids[i] == self.image_token_id).nonzero()[0].item()
            image_positions.append(pos)

        image_embeds = image_embeds.to(
            dtype=text_embeds.dtype,
            device=text_embeds.device
        )

        inputs_embeds = []
        for i, image_pos in enumerate(image_positions):
            before = text_embeds[i, :image_pos, :]
            after = text_embeds[i, image_pos + 1:, :]

            combined = torch.cat([
                before,
                image_embeds[i],
                after
            ], dim=0)

            inputs_embeds.append(combined)
        inputs_embeds = torch.stack(inputs_embeds, dim=0)

        # labels
        new_labels = []

        assistant_ids = assistant_ids.to(input_ids.device)

        for i in range(input_ids.shape[0]):

            # 当前样本
            label = input_ids[i].clone()
            label[inputs["attention_mask"][i] == 0] = -100

            # 1. 找 assistant 后答案开始的位置
            assistant_pos = None

            for j in range(len(label) - len(assistant_ids) + 1):
                if torch.equal(
                        label[j:j + len(assistant_ids)],
                        assistant_ids
                ):
                    assistant_pos = j + len(assistant_ids)
                    break

            if assistant_pos is None:
                raise ValueError("没有找到 assistant token")

            # 2. assistant 答案之前全部不计算 loss
            label[:assistant_pos] = -100

            # 3. 找当前样本 <image> 的位置
            image_pos = image_positions[i]

            before_label = label[:image_pos]
            after_label = label[image_pos + 1:]

            # 一个 <image> 替换成 196 个 image token
            # 这些位置都不计算语言 loss
            image_label = torch.full(
                (image_embeds.shape[1],),
                -100,
                dtype=label.dtype,
                device=label.device
            )

            # 4. 拼起来
            combined_label = torch.cat(
                [before_label, image_label, after_label],
                dim=0
            )

            new_labels.append(combined_label)

        #attention mask
        attention_mask = inputs["attention_mask"]
        new_attention_masks = []
        for i in range(input_ids.shape[0]):
            image_pos = image_positions[i]

            mask = attention_mask[i]

            before_mask = mask[:image_pos]
            after_mask = mask[image_pos + 1:]

            # 196 个 image token 都是有效输入，所以全部是 1
            image_mask = torch.ones(
                image_embeds.shape[1],
                dtype=mask.dtype,
                device=mask.device
            )

            combined_mask = torch.cat(
                [before_mask, image_mask, after_mask],
                dim=0
            )

            new_attention_masks.append(combined_mask)

        attention_mask = torch.stack(
            new_attention_masks,
            dim=0
        ).contiguous()

        # 5. 重新组成 batch
        labels = torch.stack(new_labels, dim=0)
        inputs_embeds = inputs_embeds.contiguous()
        labels = labels.contiguous()

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels = labels
        )
        loss = outputs.loss
        return loss

    def generate(self, images, questions):
        messages = []
        for question in questions:
            message = [
                {"role": "user", "content": question}
            ]
            messages.append(message)

        texts = []
        for message in messages:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True
        )
        device = next(self.llm.parameters()).device
        input_ids = inputs["input_ids"].to(device)

        image_tokens = self.iencoder(images)
        image_embeds = self.mm_projector(image_tokens)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = image_embeds.to(
            device=text_embeds.device,
            dtype=text_embeds.dtype
        )

        image_positions = []
        for i in range(input_ids.shape[0]):
            pos = (input_ids[i] == self.image_token_id).nonzero()[0].item()
            image_positions.append(pos)

        new_embeds = []
        for i, image_pos in enumerate(image_positions):
            before = text_embeds[i, :image_pos, :]
            after = text_embeds[i, image_pos + 1:, :]
            combined = torch.cat(
                [before, image_embeds[i], after],
                dim=0
            )
            new_embeds.append(combined)
        inputs_embeds = torch.stack(new_embeds, dim=0)

        with torch.no_grad():
            outputs = self.llm(
                inputs_embeds=inputs_embeds
            )
        logits = outputs.logits

        generated_ids = []
        for _ in range(50):
            # 取最后一个位置对“下一个 token”的预测
            next_id = logits[:, -1, :].argmax(dim=-1)

            # 遇到 EOS 就停止
            if next_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id.item())

            # token id → embedding
            next_embed = self.llm.get_input_embeddings()(next_id)
            next_embed = next_embed.unsqueeze(1)

            # 接到序列末尾
            inputs_embeds = torch.cat(
                [inputs_embeds, next_embed],
                dim=1
            )

            # 再预测下一个
            with torch.no_grad():
                outputs = self.llm(
                    inputs_embeds=inputs_embeds
                )

            logits = outputs.logits
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return answer

class LLaVADataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        question = "<image> What is in this image?"
        answer = f"A {self.class_names[label]}."

        return image, question, answer

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear
        # self.A = nn.Linear(896, 8, bias=False)
        # self.B = nn.Linear(8, 896, bias=False)
        self.A = nn.Linear(
            original_linear.in_features, r, bias=False
        )
        self.B = nn.Linear(
            r, original_linear.out_features, bias=False
        )
        for p in self.original_linear.parameters():
            p.requires_grad = False

        # 跟原 q_proj 保持一致
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        self.A = self.A.to(device=device, dtype=dtype)
        self.B = self.B.to(device=device, dtype=dtype)

        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

        self.scaling = alpha / r

    def forward(self, x):
        return (
                self.original_linear(x)
                + self.scaling * self.B(self.A(x))
        )



def train():
    lora_params = [
        p for p in model.llm.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW([
        {
            "params": model.iencoder.vision_model.encoder.layers[-1].parameters(),
            "lr": 1e-5
        },
        {
            "params": model.mm_projector.parameters(),
            "lr": 1e-4
        },
        {
            "params": lora_params,
            "lr": 1e-4
        }

    ])


    for epoch in range(epochs):
        for images, questions, answers in train_loader:
            start = time.time()
            images = images.to(device)
            optimizer.zero_grad()
            loss = model(images, questions, answers)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")
            # print("backward time:", time.time() - start)

    # torch.save(model.state_dict(), "./data/LoRA02.pth")
    lora_state = {
        name: param.cpu()
        for name, param in model.llm.state_dict().items()
        if ".A." in name or ".B." in name
    }

    torch.save(lora_state, "lora_weights.pt")
    torch.save(model.state_dict(), "data/finetuned.pth")

def load_model():
    model.load_state_dict(
        torch.load("./data/minillava.pth", map_location=device)
    )

def test():
    model.eval()

    device = next(model.parameters()).device

    total = 0
    correct = 0

    for i in range(min(eval_size, len(dataset))):
        image, question, answer = dataset[i]

        test_image = image.unsqueeze(0).to(device)
        prediction = model.generate(test_image, [question])

        if answer.strip().lower() == prediction.strip().lower():
            correct += 1
            print("正确",answer)

        total += 1

    accur = correct / total
    print("正确率：", accur)

def lora_init():
    print("LoRA Preparing...")
    for p in model.llm.parameters():
        p.requires_grad = False
    for layer in model.llm.model.layers:
        # Attention
        layer.self_attn.q_proj = LoRALinear(
            layer.self_attn.q_proj, r=8, alpha=16
        )
        layer.self_attn.v_proj = LoRALinear(
            layer.self_attn.v_proj, r=8, alpha=16
        )
        layer.self_attn.o_proj = LoRALinear(
            layer.self_attn.o_proj, r=8, alpha=16
        )

        # MLP
        layer.mlp.gate_proj = LoRALinear(
            layer.mlp.gate_proj, r=8, alpha=16
        )
        layer.mlp.up_proj = LoRALinear(
            layer.mlp.up_proj, r=8, alpha=16
        )
        layer.mlp.down_proj = LoRALinear(
            layer.mlp.down_proj, r=8, alpha=16
        )

    print(sum(
        p.numel()
        for p in model.llm.parameters()
        if p.requires_grad
    ))
    for name, p in model.llm.named_parameters():
        if p.requires_grad:
            print(name, p.shape)

def lora_load():
    print("LoRA loading...")
    lora_state = torch.load(
        "lora_weights.pt",
        map_location="cpu",
        weights_only=True
    )

    model.llm.load_state_dict(
        lora_state,
        strict=False
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MiniLLaVA with manual LoRA")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--checkpoint", default="data/LoRA02.pth")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_size < 1:
        parser.error("epochs and eval-size must be positive")
    global batch_size, epochs, train_loader, dataset, device, model, eval_size
    mode = 2
    batch_size = 4
    epochs = 4
    # images = torch.rand(batch_size,3,224,224)
    # questions = ["<image> What is in this image?", "<image> What animal?"]
    # answers = ["A cat.", "A dog."]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    cifar_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    dataset = LLaVADataset(cifar_train)
    small_dataset = Subset(dataset, range(500))
    train_loader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniLLaVA().to(device)
    print(len(small_dataset))

    epochs = args.epochs
    eval_size = args.eval_size
    lora_init()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.float()
    if args.mode == "train":
        train()
    else:
        dataset = LLaVADataset(datasets.CIFAR10(root="./data", train=False, download=True, transform=transform))
        test()

if __name__ == "__main__":
    main()
