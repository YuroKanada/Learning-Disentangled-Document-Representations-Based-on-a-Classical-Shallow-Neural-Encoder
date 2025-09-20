import pickle
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.doc2dr import Doc2DR
from trainers.trainer import Trainer
from utils.optimizers import Adam
from utils.data import create_contexts_target, to_cpu
from config.config import GPU


pkl_file = 'traindata_reviewWord500_param.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
    title_to_id = params['title_to_id']
    id_to_title = params['id_to_title']
    contexts = params['contexts']
    target = params['target']
    corpus = params['corpus']
    all_genre = params['all_genre']
    genre = params['genre']

# ハイパーパラメータの設定
#review_count = 5
hidden_size = 50
batch_size = 800
max_epoch = 10
window_size = 5


title_size = len(title_to_id) #重みのサイズを指定するときに語彙数＋タイトル数の重みになるように

#データの準備　すでにcorpus,word_to_id,id_to_wordはすでに作ってある
#-------------------------------------------------------------
vocab_size = len(word_to_id)

contexts = create_contexts_target(corpus, window_size)#データづくり

# del all_data[:window_size]
# del all_data[-window_size:] #all_dataはtargetになるので、入力として使えない前後windowsize分を削除

#target = all_data #タイトルと単語のインデックスのセットデータを入力に
#all_genre = all_genre #corpusみたいな感じ
# genre = genre

# if GPU:
#     contexts, target, genre = to_gpu(contexts), to_gpu(target), to_gpu(genre)
#-------------------------------------------------------------

# モデルなどの生成
#model = CBOW(vocab_size, hidden_size, window_size, corpus)
model = Doc2DR(vocab_size, title_size, hidden_size, window_size, corpus, all_genre)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習開始
trainer.fit(contexts, target, genre, max_epoch, batch_size)
trainer.plot(save_path=f'newinKLD_word&KLD_1d_review150_epoch{max_epoch}.png')

# 後ほど利用できるように、必要なデータを保存
word_vecs = model.word_vecs
if GPU:
    word_vecs = to_cpu(word_vecs)

# 結果を pickle に保存
model.save_stats()

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
params['title_to_id'] = title_to_id
params['id_to_title'] = id_to_title
#params['all_word'] = cut

pkl_file = f'newinKLD_word&KLD_1d_review150_{max_epoch}_param.pkl'  
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)