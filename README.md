# New Mind Social Media Analysis Task

This project is designed to analyze and match topics with related opinions and generate conclusions using a language model. The primary goal is to group topics with related opinions and generate meaningful conclusions based on these groups.

The output is not perfect and may be lack of coherence and accuracy... I tried some approaches but this project is the closest to the example conclusion that given to me. You can see the my example conclusion (my output) picture at the end of this readme file.


## Data Files

- **topics.csv**: Contains the topics data.
- **opinions.csv**: Contains the opinions data.
- **conclusions.csv**: Contains the generated conclusions data (output).

## General Explanation

- **data_loading:**  Loads the datas and also checks for the null values in topics and opinions and returns for the matching process by using pandas.
- **topic_opinion_matching:** Uses sentence_transformers to match topics with the most relevant opinions.
- **conclusion_generation:** Generate coherent conclusions based on the matched topics and opinions using GPT-2 Language Model.
- **main script:** Runs the whole process, from loading data to generating conclusions, with detailed logging at each step for debugging.
- (**Detailed Logging:**) The project includes detailed logging to help track the progress and identify any issues.

## Technologies Used

- GPT-2
- Hugging Face Transformers
- Sentence Transformers
- Pandas

### Usage

1. Place the `topics.csv` and `opinions.csv` files in the `data` directory.

2. Run the `main.py` script to perform the analysis and generate conclusions:
    ```bash
    python main.py
    ```

3. The conclusions will be saved in the `data/conclusions.csv` file.


## A part from my output: (Conclusion.csv)
![example_conclusions](https://github.com/ob06/newmind_task/assets/87376313/e47f9b70-cce0-42b3-9442-068828dc22ac)
As you can see from above, despite some of the conclusion texts may be more accurate and coherent than others. But some of them are repeating and not coherent. This shows that although my approach works and gives some coherent conclusions, it does not achieve the desired result 100%.
