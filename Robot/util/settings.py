
class Settings:

    class Setting:
        def __init__(self, name, initial, increase_callback, decrease_callback, string_callback):
            self.name = name
            self.value = initial
            self.increase_callback = increase_callback
            self.decrease_callback = decrease_callback
            self.string_callback = string_callback

        def set_value(self, value):
            self.value = value
        
        def __str__(self):
            return f"{self.name}: {self.string_callback(self.value)}"

        def increase(self):
            self.increase_callback(self)
        
        def decrease(self):
            self.decrease_callback(self)

    def __init__(self, camera_matrix, confidence):

        def increase_matrix(setting: self.Setting):
            setting.value[0, 0] = setting.value[0, 0] + 1
            setting.value[1, 1] = setting.value[1, 1] + 1
    
        def decrease_matrix(setting: self.Setting):
            setting.value[0, 0] = setting.value[0, 0] - 1
            setting.value[1, 1] = setting.value[1, 1] - 1

        #self.speed = self.Setting(speed, (lambda setting : setting.set_value(setting.value + 1)), (lambda setting : setting.set_value(setting.value - 1)), lambda value : f"{value}" )
        self.confidence = self.Setting("Confidence", confidence, (lambda setting : setting.set_value(setting.value + 0.05)), (lambda setting : setting.set_value(setting.value - 0.05)), lambda value : f"{value:.2f}" )
        self.camera_matrix = self.Setting("Intrinsic", camera_matrix, increase_matrix, decrease_matrix, lambda value : f"{value[0, 0]:.2f}")

        self.setting_selection = 0
        self.settings = [self.confidence, self.camera_matrix]
    
    def up(self):
        if self.setting_selection > 0:
            self.setting_selection -= 1
        else:
            self.setting_selection = len(self.settings) - 1            
    
    def down(self):
        self.setting_selection = (self.setting_selection + 1) % len(self.settings)
    
    def left(self):
        self.settings[self.setting_selection].decrease()
    
    def right(self):
        self.settings[self.setting_selection].increase()
        
