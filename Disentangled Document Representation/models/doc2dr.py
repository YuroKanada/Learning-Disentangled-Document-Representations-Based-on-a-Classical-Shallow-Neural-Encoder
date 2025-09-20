from models.embedding import Embedding
from models.loss import NegativeSamplingLoss, disentangled_loss
from models.scheduler import SigmoidAnnealingScheduler
import pickle
import collections
from common import np

class Doc2DR:
    def __init__(self, vocab_size, title_size, hidden_size, window_size, corpus, all_genre):
        V, T, H = vocab_size, title_size, hidden_size #titleを追加 vocab_sizeにtitle数を入れると、
        rn = np.random.randn
        self.V = V
        counts = collections.Counter()
        for genre_id in all_genre:
            counts[genre_id] += 1

        genre_size = len(counts)
        G = genre_size #ジャンルサイズだけcontexts側の重みWoutを増やす

        # 重みの初期化
        W_in = 0.1 * rn(V+T, H).astype('f') #語彙数＋タイトル数の重み行列
        W_out = 0.1 * rn(V+G, H).astype('f')#語彙数＋ジャンル数
        
        # レイヤの生成
        self.in_layers = []
        for i in range(2):
            layer = Embedding(W_in)  # Embeddingレイヤを使用
            self.in_layers.append(layer)#単語用、映画用のembレイヤを用意し、CBOWのような入力に

        self.loss_layers = []
       
        for i in range(2 * window_size): 
            layer = NegativeSamplingLoss(W_out, corpus, all_genre,V, power=0.75, sample_size=5) #negative_sampling_lossレイヤの追加
            self.loss_layers.append(layer)#↑ここでWoutを指定しているからcontext用の重み行列でembできる
        
        #genre用のNSLレイヤを追加
        layer = NegativeSamplingLoss(W_out, corpus,all_genre,V,  power=0.75, sample_size=5) #negative_sampling_lossレイヤの追加
        self.loss_layers.append(layer)#↑ここでWoutを指定しているからcontext用の重み行列でembできる    

        # すべての重みと勾配をリストにまとめる
        layers = self.in_layers + self.loss_layers #どちらもリストになっているので[]はいらない
        
        #self.disentangled_loss_layer = []
        #layer = disentangled_loss
        #self.disentangled_loss_layer.append(layer)
        self.disentangled_loss_layer = disentangled_loss(W_in)
        
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # disentangled_loss_layer のパラメータと勾配を追加
        self.params += self.disentangled_loss_layer.embed.params
        self.grads += self.disentangled_loss_layer.embed.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in
        self.det_cov_p_list = []
        self.loss_list = []
        #self.word_vecs_list = [(self.word_vecs[0],self.word_vecs[-1])]
    #-----------------------------------------------------------------------------------------------------
    #ロスをKLDのみにする
    def forward(self, contexts, target, genre,t):#順伝搬 ここで受け取るのは小規模なインデックス集合
        
        #self.word_vecs_list.append((self.word_vecs[0],self.word_vecs[-1]))
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(target[:, i]) #embメソッドのforwardに順伝搬しを獲得、バッチサイズ分あるtarget（中心単語と映画のセット）の単語のみ、映画のみでEMBに入れていく
            #print('これがtargetの中身',target[:10])
            #h2はいくつかの単語の分散表現を抽出した行列となり、h1は映画の分散表現行列となる
            #h=h1+h2なので、行数、列数は変化なし
        h *= 1 / len(self.in_layers)
        #映画と単語のベクトルを平均にしたベクトルを利用
        
        loss = 0
        losw = 0
        lossw = 0
        lossg = 0
        kld_loss = 0
        #weight = 0
        #SASの実装-------------------------------------
        change_weight = SigmoidAnnealingScheduler()
        weight = change_weight.get_value(t)#kld項の重みを変える
        print('重み',weight)
        #-----------------------------------------------
        
        #単語パターン予測
        for i, layer in enumerate(self.loss_layers[:-1]):#loss_layerの数（周辺単語数+genre用）になっているので、最後のレイヤだけを無視する
            lossw = layer.forward(h, contexts[:, i])
            losw += lossw
            loss += lossw#contextsの一つの要素（idx）を順伝搬、ここでジャンル追加
            #print(f'単語のパターン予測{i}回目のロス:{losss}')
            
        #genreのロスを計算
        #lossg = self.loss_layers[-1].genreforward(h, genre, self.V)
        
        loss += lossg
        #print(f'ジャンル予測のロス:{losss}')
        
        #================================================================================================
        

        kld_loss = self.disentangled_loss_layer.forward(target[:, 0])#バッチサイズ分の映画タイトルベクトル集合
        
        # kld_loss = self.disentangled_loss_layer.forward(target[:, 1])#バッチサイズ分の単語ベクトル集合
        
        #==================================================================================================
            
        self.loss_list.append((losw, lossg, kld_loss, weight))
        
        #print(f'KLDのロス:{kld_loss}')
        loss += kld_loss * weight
        self.weight = weight
 
        return loss
    #-------------------------------------------------------------------------
    
    #------------------------------------------------------------------------  
    def backward(self, dout=1):

        dhw = 0 #単語パターン予測の勾配
        dhg = 0 #ジャンル予測の勾配
        dh = 0
        dkld = 0 #KLD算出項の勾配
        dweight = self.weight
        #単語パターン予測とジャンル予測の勾配を別々で求めて、その和を伝搬する
        
        #単語パターン予測
        for i, layer in enumerate(self.loss_layers[:-1]):
            dhw += layer.backward(dout)
        
        #ジャンル予測
        #dhg = self.loss_layers[-1].backward(dout)
            
        #異なる学習タスクの勾配を足し合わせる
        dh = dhg + dhw
        
        # #入力の単語、映画のembレイヤに均等に重みを伝搬
        # dh *= 1 / len(self.in_layers)
        
        if dweight > 0:
            #KLD項への逆伝搬
            dkld = self.disentangled_loss_layer.backward(dout) * dweight
        else:
            dkld = 0
        
        #単語のEmbレイヤにはタスクの勾配を伝搬
        self.in_layers[1].backward(dh)
        
        #映画のEMBレイヤにはタスクの勾配とKLDの勾配を伝搬
        self.in_layers[0].backward(dh + dkld)
        
        #=============================================
        # #映画のEmbレイヤにはタスクの勾配を伝搬
        # self.in_layers[0].backward(dh)
        
        # #単語のEMBレイヤにはタスクの勾配とKLDの勾配を伝搬
        # self.in_layers[1].backward(dh + dkld)
        #===============================================

        
        return None
    
    def save_stats(self, filename='all_loss_newinKLD_1d_word&KLD_review150.pkl'):
        # det_cov_p と KLD を辞書にまとめて保存
        data = {
            #'det_cov_p': self.det_cov_p_list,
            'all_loss': self.loss_list,
            #'word_vecs_list':self.word_vecs_list
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved stats to {filename}')