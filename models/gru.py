import tensorflow as tf

class GRUModel(tf.keras.Model):

    def __init__(self, num_ids, embedding_dim, gru_units_1, gru_units_2, dense_units, dropout):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_ids,
            output_dim=embedding_dim
        )

        self.gru_1 = tf.keras.layers.GRU(
            gru_units_1,
            return_sequences=True
        )

        self.dropout_1 = tf.keras.layers.Dropout(dropout)

        self.gru_2 = tf.keras.layers.GRU(
            gru_units_2,
            return_sequences=False
        )

        self.dropout_2 = tf.keras.layers.Dropout(dropout)

        self.dense = tf.keras.layers.Dense(
            dense_units,
            activation="relu"
        )

        self.dropout_3 = tf.keras.layers.Dropout(dropout)

        self.output_layer = tf.keras.layers.Dense(
            num_ids,
            activation="softmax"
        )
    
    def call(self, inputs, training=False):
        # Create embedding
        x = self.embedding(inputs)

        x = self.gru_1(x)
        x = self.dropout_1(x, training=training)

        x = self.gru_2(x)
        x = self.dropout_2(x, training=training)

        x = self.dense(x)
        x = self.dropout_3(x, training=training)

        # Output of Dense Layer
        return self.output_layer(x)
        
