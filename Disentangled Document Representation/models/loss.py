from common import np
from utils.sampler import UnigramSampler, genreSampler
from utils.math_utils import cross_entropy_error, kl_divergence, kl_divergence_1d_grad
from models.embedding import Embedding, EmbeddingDot


class NegativeSamplingLoss:

#単語予測に関するLossを計算するレイヤ
#ジャンル予測についても同様な計算方法なので、genreforwardという関数を追加
#逆伝搬は同じなのでbackwardは一つで良い
    
    def __init__(self, W, corpus, all_genre,vocab_size,power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.genre_sampler = genreSampler(all_genre, power, sample_size, vocab_size)#genreのためのサンプリングレイヤ
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):# hは入力の中間表現（targetの単語ベクトル）、ここに書いてあるtargetは実際はコンテキストが格納される
        batch_size = target.shape[0]
        negative_word = self.sampler.get_negative_sample(target)#contextsをサンプリング

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)#embed_dotレイヤの一つ目を正例に使う
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_word[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
       
        return loss

    
    #ジャンルのロスを計算するためのレイヤ    
    def genreforward(self, h, genre ,V):# hは入力の中間表現（targetの単語ベクトル）、ここに書いてあるtargetは実際はコンテキストが格納される
        batch_size = genre.shape[0]
        negative_genre = self.genre_sampler.genre_negative_sample(genre, V)#genreをサンプリング

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, genre)#embed_dotレイヤの一つ目を正例に使う
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_genre[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label) # if target[0] >= len(self.V): #ジャンルの場合

       
        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)#sigmoidwithlossにdoutをいれる
            dh += l1.backward(dscore)#embed_dotレイヤに入れる

        return dh


class disentangled_loss:

#ベクトルの各次元の独立化のためのKLDによる正則化項を計算するレイヤ
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params  # パラメータの管理
        self.grads = self.embed.grads    # 勾配の管理
        #self.params, self.grads = [], []
 
        #self.loss = None
        #self.t_index = None
    
    def forward(self, t_index):
        #self.t_index = t_index
        #print('映画のidx',t_index)
        vt = self.embed.forward(t_index)#.copy()#コピーを使うことでvtの値の変化により元のparamが変化することがない
        batch_size = vt.shape[0]#映画タイトルの数
        dimension_size = vt.shape[1]
        #print('KLDを計算するベクトル行列の形',vt.shape)
        mean_p = np.mean(vt, axis=0)
        # データを平均ベクトルで中心化
        centered_data = vt - mean_p
        
        #----------------------------------
        #------------------------------------------------------------------
        var_p = np.var(vt, axis=0, ddof=0)
        KLD = kl_divergence(mean_p, var_p)
        
        self.vt = vt
        self.mean_p = mean_p
        #self.cov_p = cov_p
        self.var_p = var_p
        self.centered_data = centered_data
        
        return KLD
    
    
    def backward(self, dout=1):
        batch_size = self.vt.shape[0]
        dimension_size = self.vt.shape[1]

        # 各次元独立な勾配を計算
        dmean_p, dvar_p = kl_divergence_1d_grad(self.mean_p, self.var_p)

        # mean_p による勾配
        #dmean_p = -dmean_p  # 正則化項の符号変更
        dvt_mean = np.ones((batch_size, dimension_size)) * dmean_p / batch_size

        # var_p による勾配
        #dvar_p = -dvar_p  # 正則化項の符号変更
        dvt_var = 2 * self.centered_data * dvar_p / batch_size

        # 合計して dvt を求める
        dvt = (dvt_mean + dvt_var) * dout
        self.embed.backward(dvt)

        return dvt

class SigmoidWithLoss:

#sigmoidと交差エントロピー誤差をまとめたレイヤ

    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ
    
    def forward(self, x, t):#順伝搬のラスト　xはscore,tはゼロの並んだ配列
        #h,negative(positive)なcontextsを入れてsocreを一つ一つに算出
        self.t = t
        self.y = 1 / (1 + np.exp(-x))#1/1+log(-x)がyに（sigmoidの出力）

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):#逆伝搬の最初
        
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx