import pandas as pd
from langdetect import detect, DetectorFactory
from textblob import TextBlob

# Set seed for reproducibility
DetectorFactory.seed = 0

# ============================================================================
# SAFE LANGUAGE DETECTION
# ============================================================================

def safe_detect_language(text):
    """Safely detect language with error handling."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'unknown'
    
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        # Handle: empty, too short, can't detect, etc.
        return 'unknown'

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

def get_sentiment(text):
    """Analyze sentiment with error handling."""
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', round(polarity, 3)
        elif polarity < -0.1:
            return 'negative', round(polarity, 3)
        else:
            return 'neutral', round(polarity, 3)
    except Exception as e:
        return 'error', None

# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Read CSV
print("📖 Reading CSV...")
df = pd.read_csv('reviews.csv')

#df = df.head(1000) 
print(f"   Original shape: {df.shape}")

# Drop NA in comments
print("\n🧹 Dropping NA values in comments...")
df = df.dropna(subset=['comments'])
print(f"   New shape: {df.shape}")
print(f"   Rows dropped: {891964 - len(df)}")

# Remove rows with empty or whitespace-only comments
print("\n🧹 Removing empty/whitespace comments...")
before = len(df)
df = df[df['comments'].str.strip().str.len() > 0]
print(f"   Additional rows removed: {before - len(df)}")

# Language detection
print("\n🌍 Detecting languages...")
df['language'] = df['comments'].apply(safe_detect_language)

# Show progress
print(f"   ✓ Languages detected")
print(f"   Top languages:")
for lang, count in df['language'].value_counts().head(10).items():
    pct = (count / len(df) * 100)
    print(f"      {lang}: {count} ({pct:.1f}%)")

# Sentiment analysis
print("\n😊 Analyzing sentiment...")
sentiment_data = df['comments'].apply(get_sentiment)
df['sentiment'] = sentiment_data.apply(lambda x: x[0])
df['polarity'] = sentiment_data.apply(lambda x: x[1])

# Show results
print(f"   ✓ Sentiment analyzed")
print(f"\n   Sentiment Distribution:")
for sentiment, count in df['sentiment'].value_counts().items():
    pct = (count / len(df) * 100)
    print(f"      {sentiment.upper()}: {count} ({pct:.1f}%)")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)

# Sentiment by language (top 5 languages)
print("\n📊 Sentiment by Language (Top 5):")
top_langs = df['language'].value_counts().head(5).index

for lang in top_langs:
    lang_df = df[df['language'] == lang]
    print(f"\n   {lang.upper()} ({len(lang_df)} reviews):")
    
    sentiment_dist = lang_df['sentiment'].value_counts()
    for sentiment, count in sentiment_dist.items():
        pct = (count / len(lang_df) * 100)
        print(f"      {sentiment}: {count} ({pct:.1f}%)")

# Polarity statistics
print("\n📈 Polarity Statistics:")
print(f"   Mean polarity: {df['polarity'].mean():.3f}")
print(f"   Median polarity: {df['polarity'].median():.3f}")
print(f"   Std deviation: {df['polarity'].std():.3f}")
print(f"   Min: {df['polarity'].min():.3f}")
print(f"   Max: {df['polarity'].max():.3f}")

# Error/Unknown handling
error_count = len(df[df['sentiment'] == 'error'])
unknown_count = len(df[df['language'] == 'unknown'])
print(f"\n⚠️  Data Quality:")
print(f"   Sentiment errors: {error_count}")
print(f"   Unknown languages: {unknown_count}")

# ============================================================================
# RESULTS
# ============================================================================

# Display sample results
print("\n" + "="*70)
print("SAMPLE RESULTS (first 20)")
print("="*70)
sample = df[['comments', 'language', 'sentiment', 'polarity']].head(20)
for idx, row in sample.iterrows():
    comment = row['comments'][:60] + "..." if len(str(row['comments'])) > 60 else row['comments']
    print(f"\n   [{idx}] {comment}")
    print(f"       Language: {row['language']} | Sentiment: {row['sentiment']} | Polarity: {row['polarity']}")

# Save results
print("\n" + "="*70)
output_file = 'sentiment_results.csv'
df.to_csv(output_file, index=False)
print(f"✅ Results saved to: {output_file}")
print(f"   Total rows: {len(df)}")
