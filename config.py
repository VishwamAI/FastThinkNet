class Config:
    """Configuration settings for the FastThinkNet project."""

    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10
        self.hidden_layers = [64, 64]
        self.activation = "relu"
        self.optimizer = "adam"

    def __str__(self):
        return (
            f"Config(learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, "
            f"num_epochs={self.num_epochs}, "
            f"hidden_layers={self.hidden_layers}, "
            f"activation='{self.activation}', "
            f"optimizer='{self.optimizer}')"
        )
