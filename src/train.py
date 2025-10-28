import torch
import os
import argparse
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from src.data_processing import get_dataloaders
from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Fine-tune a T5 model for Text-to-SQL.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="The initial learning rate.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="The name of the pretrained model to use.")

    args = parser.parse_args()
    print("Arguments parsed successfully.")
    
    writer = SummaryWriter()

    print(f"loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("loading pre-trained model.")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)
   
    print("creating DataLoaders.")
    max_token_length = tokenizer.model_max_length
    train_dataloader, val_dataloader, _ = get_dataloaders('data/train.csv','data/validation.csv', 'data/test.csv', args.batch_size ,tokenizer, max_token_length )
    
    
    total_training_steps = len(train_dataloader) * args.num_epochs
    optimizer = optim.AdamW(model.parameters(), lr= args.learning_rate)
    warmup_steps =int( total_training_steps * 0.07)
    scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_training_steps )

    print("\nModel and data loaded successfully. Ready for training.")

    output_dir = "model_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0


    #EPOCH LOOP

    for epoch in range(args.num_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            for key, value in batch.items():
                batch[key] = value.to(device)
            optimizer.zero_grad()    
            outputs= model(**batch)
            loss = outputs.loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}, Train Loss: {loss.item()}")
    
        model.eval()
        total_val_loss= 0
        with torch.no_grad():
            for batch in val_dataloader:
                for key, value in batch.items():
                    batch[key] = value.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss/len(val_dataloader)
        writer.add_scalar('Loss/validation', avg_val_loss, global_step)
        torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss:{avg_val_loss}")

    writer.close()


if __name__ == "__main__":
    main()