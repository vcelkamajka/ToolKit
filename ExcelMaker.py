import pandas as pd

def ExcelReport(df, name, file=None):

    if file is not None:
        df = pd.read_csv(file)
    if file is None:
        df = df

    with pd.ExcelWriter(name+'.xlsx', engine="xlsxwriter") as writer:
        workbook = writer.book
        ws = workbook.add_worksheet("Sheet1")

        font = {"font_name": "Times New Roman", "font_size": 11}

        def make(**kw):
            return workbook.add_format({**font, **kw})

        # Header: line above and below
        header_fmt = make(bold=True, align="center", top=1, bottom=1)

        # Middle rows: no borders
        mid_text = make()
        mid_num = make(num_format="#,##0.0??", align="right")

        # Last row: line below only
        last_text = make(bottom=1)
        last_num = make(num_format="#,##0.0??", align="right", bottom=1)

        numeric = [pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]

        # Header row
        for c, col in enumerate(df.columns):
            ws.write(0, c, col, header_fmt)

        # Data rows
        last_row = len(df)
        for r, row in enumerate(df.itertuples(index=False), start=1):
            for c, value in enumerate(row):
                if r == last_row:
                    fmt = last_num if numeric[c] else last_text
                else:
                    fmt = mid_num if numeric[c] else mid_text
                ws.write(r, c, value, fmt)

        ws.set_column(0, len(df.columns) - 1, 15)
