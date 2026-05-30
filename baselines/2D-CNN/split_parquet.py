import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.model_selection import train_test_split


parquet_file = r'E:\projects\datasets\Kitsune\data_kitsune.parquet'
table = pq.read_table(parquet_file).to_pandas()

# 切分数据集为预训练集,训练集,验证集, 比例约为8:1:1
train_df, test_df = train_test_split(table, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 可选择将切分后的数据保存为新的 Parquet 文件
train_table = pa.Table.from_pandas(train_df)
test_table = pa.Table.from_pandas(test_df)
val_table = pa.Table.from_pandas(val_df)

train_parquet_file = 'train_data.parquet'
test_parquet_file = 'test_data.parquet'
val_parquet_file = 'val_data.parquet'

pq.write_table(train_table, train_parquet_file)
pq.write_table(test_table, test_parquet_file)
pq.write_table(val_table, val_parquet_file)
