# my_model.py
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Add, MultiHeadAttention

# 定义残差块
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, kernel_size, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

# 定义注意力块
def attention_block(x, filters):
    x = MultiHeadAttention(num_heads=4, key_dim=filters)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# 定义自定义损失函数
def custom_loss(y_true, y_pred):
    """
    自定义损失函数，在传统二元交叉熵基础上加入置信度加权。
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    返回值:
    - 加权后的平均损失
    """
    confidence = tf.abs(y_pred - 0.5) * 2  # 计算置信度
    base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # 计算基础损失
    weighted_loss = base_loss * confidence  # 加权损失
    return tf.reduce_mean(weighted_loss)  # 返回加权后的平均损失

# 创建复杂模型
def create_complex_model(input_shape=(1000, 4)):
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 添加多个残差块和注意力块
    for _ in range(3):
        x = residual_block(x, filters=64)
        x = attention_block(x, num_heads=4, key_dim=64)
    
    x = MaxPooling1D(pool_size=2)(x)
    
    for _ in range(3):
        x = residual_block(x, filters=128)
        x = attention_block(x, num_heads=4, key_dim=128)
    
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    return model

