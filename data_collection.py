import os

from utils import load_from_file, save_to_file


def aggregate_responses(raw_fpath, result_fpath):
    """
    :param raw_fpath: file path to raw csv containing Google form responses
    :param result_fpath: file path to aggregated results
    Assume fields of csv file are Timestamp, Username, LTL template type, Number of Propositions, Utterance.

    The raw responses from Google form have many empty columns in each row.
    To aggregate responses and save them in a csv file,
    1. Download the responses to the Google form as a csv file then place it in the data folder.
    2. run python data_collection.py
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


if __name__ == '__main__':
    raw_fpath = os.path.join("data", "raw_responses.csv")
    result_fpath = os.path.join("data", "aggregated_responses.csv")
    aggregate_responses(raw_fpath, result_fpath)
