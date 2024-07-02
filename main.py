import logging
import pandas as pd
from data_loading import load_data
from topic_opinion_matching import match_topics_to_opinions
from conclusion_generation import generate_conclusions, save_conclusions

# Logging for easier debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the main script.")

    # Loading the data
    logging.info("Loading data...")
    topics_df, opinions_df = load_data('data/topics.csv', 'data/opinions.csv')
    logging.info(f"Loaded {len(topics_df)} topics and {len(opinions_df)} opinions.")

    # Not used all the data to speed up the process
    topics_df = topics_df.head(100)  # Use the first 100 topics
    opinions_df = opinions_df.head(500)  # Use the first 500 opinions

    # Matching opinions with topics
    logging.info("Matching topics to opinions...")
    matched_opinions = match_topics_to_opinions(topics_df, opinions_df)
    logging.info("Matching completed.")

    # Generating conclusions
    logging.info("Generating conclusions...")
    conclusions = generate_conclusions(matched_opinions)
    logging.info("Conclusions generated.")

    # Saving conclusions
    logging.info("Saving conclusions to CSV...")
    save_conclusions(conclusions, 'data/conclusions.csv')
    logging.info("Conclusions saved successfully.")

    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()
