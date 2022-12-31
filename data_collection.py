import os
from collections import defaultdict

from utils import load_from_file, save_to_file


def aggregate_responses(raw_fpath, result_fpath):
    """
    :param raw_fpath: file path to raw csv containing Google form responses
    :param result_fpath: file path to aggregated results
    Assume fields of csv file are Timestamp, Username, LTL template type, Number of Propositions, Utterance.

    The raw responses from Google form have many empty columns in each row.
    To aggregate responses and save them in a csv file,
    1. Fix incorrect responses in Google Sheets if needed.
    2. Download the responses to the Google form from Google Sheets as a csv file then place it in the data folder.
    3. run python data_collection.py
    """
    raw_data = load_from_file(raw_fpath, noheader=False)
    fields = raw_data.pop(0)
    result_fields = fields[:4] + ["Utterance"]
    results = [result_fields]

    for row in raw_data:
        result_row = [None] * len(result_fields)
        for col_idx, (field, col) in enumerate(zip(fields, row)):
            if field == "Number of Propositions":
                if col:
                    result_row[3] = col
            elif "Utterance" in field:
                if col:
                    if result_row[-1]:  # more than 1 utterances recorded for this participant
                        results.append(result_row)
                        result_row = result_row[:]  # start a new row
                        result_row[-1] = col  # every col same as last row except utterance
                    else:  # 1st utterance recorded for this participant
                        result_row[-1] = col
            else:
                result_row[col_idx] = col
        results.append(result_row)
    save_to_file(results, result_fpath)


def analyze_responses(result_fpath, analysis_fpath):
    """
    :param result_fpath: file path to aggregated results
    :param analysis_fpath: path to file containing analysis of the results
    """
    results = load_from_file(result_fpath, noheader=True)
    counter = defaultdict(lambda: defaultdict(int))

    for _, _, ltl_type, nprops, _ in results:
        counter[ltl_type][nprops] += 1

    analysis = [["LTL Template Type", "Number of Propositions", "Number of Utterances"]]
    for ltl_type, nprops2count in counter.items():
        for nprops, count in nprops2count.items():
            analysis.append([ltl_type, nprops, count])
    save_to_file(analysis, analysis_fpath)


if __name__ == '__main__':
    raw_fpath = os.path.join("data", "raw_responses_batch1.csv")
    result_fpath = os.path.join("data", "aggregated_responses_batch1.csv")
    analysis_fpath = os.path.join("data", "analysis_batch1.csv")
    # aggregate_responses(raw_fpath, result_fpath)
    analyze_responses(result_fpath, analysis_fpath)
