import sqlite3 as sq
import secrets
import time
from collections import defaultdict
import random
from tabulate import tabulate
from fastargs.decorators import param, section


def update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_training_samples, loss_bin_l, loss_bin_u, test_acc, train_time, perfect_model_count, tested_model_count, save_path, status):
    return f"""
    REPLACE INTO model_stats VALUES ( '{model_id}', {data_seed}, {training_seed}, {num_training_samples}, {loss_bin_l}, {loss_bin_u}, {test_acc}, {train_time}, {perfect_model_count}, {tested_model_count}, '{save_path}', '{status}' )"""

def create_model_stats_table_sql_script():
    return f"""CREATE TABLE IF NOT EXISTS model_stats (
	model_id TEXT PRIMARY KEY,
    data_seed INTEGER,
    training_seed INTEGER,
    num_train_samples INTEGER,
    loss_bin_l REAL,
    loss_bin_u REAL,
    test_acc REAL,
    train_time REAL,
    perfect_model_count INTEGER,
    tested_model_count INTEGER,
    save_path TEXT,
    status TEXT);"""

def create_model_stats_table(db_path):
    con = sq.connect(db_path, isolation_level="EXCLUSIVE")
    con.execute(create_model_stats_table_sql_script())
    con.commit()
    con.close()


def update_model_stats_table(db_path, model_id, data_seed, training_seed, num_training_samples, loss_bin_l, loss_bin_u, test_acc, train_time, perfect_model_count, tested_model_count, save_path, status):
    con = sq.connect(db_path)
    cur = con.cursor()
    cur.execute(
        update_model_stats_table_sql_script(
            model_id, data_seed, training_seed, num_training_samples, 
            loss_bin_l, loss_bin_u, test_acc, train_time, perfect_model_count, tested_model_count, save_path, status)
        )
    # update model_stats table
    con.commit()
    con.close()

def get_model_stats_summary_sql_script():
    return """
    SELECT
        num_train_samples, loss_bin_l, loss_bin_u,
        SUM(perfect_model_count),
        AVG(test_acc),
        AVG(train_time)
    FROM
        model_stats
    WHERE 
        status = 'COMPLETE'
    GROUP BY
        num_train_samples, loss_bin_l, loss_bin_u
    ;"""

def get_model_stats_summary(db_path, verbose=True):
    con = sq.connect(db_path)
    rows = con.execute(
        get_model_stats_summary_sql_script()
    ).fetchall()
    if verbose:
        print(tabulate(rows, headers=['num_train_samples', 'loss_bin_l', "loss_bin_u", "SUM(perfect_model_count)", "AVG(test_acc)", "AVG(train_time)"], tablefmt='psql'))
    con.close()
    return rows

def get_model_stats(db_path):
    con = sq.connect(db_path)
    rows = con.execute("SELECT * FROM model_stats").fetchall()
    con.close()
    return rows

@section('distributed')
@param('training_seed')
@param('data_seed')
def get_next_config(db_path, loss_bins, num_samples, training_seed=None, data_seed=None):
    # this function has to figure out the loss bins, num_samples, and data_seed it need to select
    # loss bins and num samples will depend on two things
    # 1. the number of models within the combination of loss bins and num samples with status complete
    # 2. the latest tested data_seed
    # we will focus on the first one
    # we will first query a summary table from the model stats table
    # we will then find the combination with the lowest number of models
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    rows = con.execute(get_model_stats_summary_sql_script()).fetchall()
    model_cnt_dict = defaultdict(int)
    for row in rows:
        model_cnt_dict[(row[0], row[1], row[2])] = row[3]
    min_cnt = float('inf')
    for loss_bin in loss_bins:
        for num_sample in num_samples:
            model_cnt = model_cnt_dict[(num_sample, loss_bin[0], loss_bin[1])]
            if model_cnt < min_cnt:
                min_cnt = model_cnt
                next_loss_bin = loss_bin
                next_num_sample = num_sample
    rows = con.execute("""
    SELECT
        MAX(data_seed),
        MAX(training_seed)
    FROM
        model_stats
    ;""")
    data_seed_next, training_seed_next = rows.fetchone()

    loss_bin_l, loss_bin_u = next_loss_bin
    model_id = secrets.token_hex(8)
    if data_seed is not None:
        data_seed_next = data_seed
    else:
        data_seed_next = 100 if data_seed_next is None else data_seed_next+1
    if training_seed is not None:
        training_seed_next = training_seed
    else:
        training_seed_next = 200 if training_seed_next is None else training_seed_next+1
    num_train_samples = next_num_sample
    test_acc = -999
    train_time = -999
    tested_model_count = -999
    perfect_model_count = -999
    
    save_path = ""
    status = "PENDING"
    
    con.execute(
        update_model_stats_table_sql_script(
            model_id, data_seed_next, training_seed_next, num_train_samples, 
            loss_bin_l, loss_bin_u, test_acc, train_time, perfect_model_count, tested_model_count, save_path, status)
        )
    con.commit()
    con.close()
    return model_id, next_loss_bin, next_num_sample, data_seed_next, training_seed_next, min_cnt


    
if __name__ == "__main__":
    from fastargs import Section, Param

    Section("distributed").params(
        loss_thres=Param(str, default="0.3,0.4,0.5"),
        num_samples=Param(str, default="2,4,8"),
        excluded_cells=Param(str, default="", desc='ex: 32_(0.3, 0.35)/16_(0.3, 0.35)'),
        target_model_count_subrun=Param(int, default=1),
        training_seed=Param(int, default=None, desc='If there is no training seed, then the training seed increment with every new runs'),
        data_seed=Param(int, default=None, desc='If there is no data seed, then the training seed increment with every new runs, otherwise, it is fix')
    )
    db_path = "tutorial.db"
    create_model_stats_table(db_path)

    loss_bins = [(0.1, 0.2), (0.2, 0.3)]
    num_samples = [16, 32]
    for i in range(10):
        print(i)
        next_config = get_next_config(db_path, [(0.1, 0.2), (0.2, 0.3)], [16, 32])
        model_id, (loss_bin_l, loss_bin_u), num_training_samples, data_seed, training_seed, _ = next_config
        print("next config: ",next_config)

        # simulated training steps
        test_acc =  random.random()
        train_time = random.random()*1000
        tested_model_count = int(random.random()*200)
        save_path = f"path/to/model_{data_seed}_{training_seed}_{loss_bin_l},{loss_bin_u}_{num_training_samples}"
        status = "COMPLETE"
        update_model_stats_table(db_path, model_id, data_seed, training_seed, num_training_samples, 1,
        loss_bin_l, loss_bin_u, test_acc, train_time, tested_model_count, save_path, status)
        

    rows = get_model_stats(db_path)
    unique_data_seed = set()
    for row in rows:
        if row[1] in unique_data_seed:
            print("duplicate data seed!", row[1])
        else:
            unique_data_seed.add(row[1])
    print("program finished")
    time.sleep(10)
