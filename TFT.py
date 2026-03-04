import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch.tuner import Tuner
from stockagent.utils.sql import SQLData


training_cutoff = "20150101"
val_cutoff = "20240101"
target = "future_avg_return"
sqldata = SQLData()
sqldata.connect()
code2id_sql = "SELECT stock_id, stock_code FROM stock_codes;"
query_sql = "SELECT * FROM stock_price_data_rm WHERE date < ?"
val_sql = "SELECT * FROM stock_price_data_rm WHERE date >= ?"
train_data = pd.read_sql(query_sql, sqldata.conn, params=(training_cutoff, ))
val_data = pd.read_sql(val_sql, sqldata.conn, params=(val_cutoff, ))
code2id = pd.read_sql(code2id_sql, sqldata.conn)
sqldata.close()

train_data = pd.merge(train_data, code2id, how = "left", on = "stock_id").dropna(axis = 0)
##update code2id
code2id = train_data.loc[:, ["stock_id", "stock_code"]].drop_duplicates()
val_data = pd.merge(val_data, code2id, how="left", on = "stock_id").dropna(axis = 0)

max_encoder_length = 21
max_prediction_length = 6
train_data = train_data.sort_values(["stock_code", "date"])
train_data = train_data.drop(["data_id", "stock_id", "date"], axis = 1)
train_data = train_data.astype({"up_limit": int, "down_limit": int}).astype({"up_limit": str, "down_limit": str})

train_data["time_idx"] = train_data.groupby("stock_code").cumcount()
val_data = val_data.sort_values(["stock_code", "date"])
val_data = val_data.drop(["data_id", "stock_id", "date"], axis = 1)
val_data["time_idx"] = val_data.groupby("stock_code").cumcount()
val_data = val_data.astype({"up_limit": int, "down_limit": int}).astype({"up_limit": str, "down_limit": str})

pred_reals = ["Log_profit", "future_avg_return", "vol_ma5", "vol_ratio"]
features = train_data.columns.to_list()
features.remove("stock_code")
features.remove("up_limit")
features.remove("down_limit")
for i in pred_reals:
    features.remove(i)

train_dataset = TimeSeriesDataSet(train_data,
                                  time_idx="time_idx",
                                  target = target,
                                  static_categoricals= ["stock_code"],
                                  group_ids=["stock_code"],
                                  min_encoder_length = max_encoder_length // 2,
                                  max_encoder_length = max_encoder_length,
                                  min_prediction_length=1,
                                  max_prediction_length=1,
                                  time_varying_known_categoricals=["up_limit", "down_limit"],
                                  time_varying_known_reals = features,
                                  time_varying_unknown_reals= pred_reals,
                                  target_normalizer=GroupNormalizer(groups=["stock_code"]),
                                  add_relative_time_idx=True,
                                  add_target_scales=True,
                                  add_encoder_length=True,
                                  allow_missing_timesteps=True,
                                  )
valid_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_data, stop_randomization=True, predict=True)
del train_data, val_data

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=20, 
    devices = "auto",
    enable_model_summary = True,
    accelerator="auto",  # run on CPU, if on multiple GPUs, use strategy="ddp"
    gradient_clip_val=0.1,
    limit_train_batches=128,
    limit_val_batches=256,
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)
tft = TemporalFusionTransformer.from_dataset(
    # dataset
    train_dataset,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=4,
    # loss metric to optimize
    loss=QuantileLoss(),
    # logging frequency
    log_interval=2,
    # optimizer parameters
    learning_rate=0.05,
    output_size = 7,
    reduce_on_plateau_patience=4
)

batch_size = 128
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=2, persistent_workers=True, shuffle = False)
val_dataloader = valid_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2, persistent_workers=True, shuffle = False)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)