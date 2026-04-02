import pandas as pd

def generate_mock_data(output_path="Cleaned_Data.csv"):
    data = [
        # Ham (Label 0)
        ("Hey, are we still meeting for lunch today?", 0),
        ("Can you please send me the report by end of day?", 0),
        ("The weather is nice today, let's go for a walk.", 0),
        ("Thanks for the help yesterday!", 0),
        ("I'll be a bit late for the meeting.", 0),
        ("Please find the attached document for your review.", 0),
        ("Don't forget to pick up the groceries on your way home.", 0),
        ("I'm looking forward to our weekend trip.", 0),
        ("Can we reschedule our call for tomorrow?", 0),
        ("The project deadline is approaching fast.", 0),
        
        # Spam (Label 1)
        ("WINNER! You have won $1,000,000 in our lottery! Click here to claim.", 1),
        ("Get cheap medicines online now! Discounted prices for limited time.", 1),
        ("URGENT: Your account has been compromised. Verify your details here.", 1),
        ("Lose weight fast with this one simple trick! Try it today.", 1),
        ("Double your income in just 5 days! Join our high-earning program.", 1),
        ("Congratulations! You've been selected for a free gift card.", 1),
        ("Make money from home with no experience needed. Sign up now!", 1),
        ("Best prices on luxury watches! Limited stock available.", 1),
        ("You have an unclaimed inheritance! Contact us immediately to receive.", 1),
        ("Claim your free reward now at www.spam-reward.com", 1)
    ]
    
    # Repeat data to ensure we have enough for 80/20 split and models can train decently
    data = data * 20 
    
    df = pd.DataFrame(data, columns=["Email", "Label"])
    df.to_csv(output_path, index=False)
    print(f"Generated mock Cleaned_Data.csv with {len(df)} entries.")

if __name__ == "__main__":
    generate_mock_data()
