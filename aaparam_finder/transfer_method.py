class TransferMethod():
    """defines the parameter transfer method
    """ 

    def __init__(self, microscope : (str|None) = None, specimen : (str|None) = None):
        """initialize parameter transfer method

        Args:
            microscope (str | None, optional): microscope type, can be confocal, widefield or twophoton. Defaults to None.
            specimen (str | None, optional): specimen, can be actin, nucleus or mito. Defaults to None.
        """

        if microscope is not None and specimen is not None:
            self.group_based = True
            self.image_based = False
            self.group_type = "mic_obj_group"
            self.group_name = f"{specimen}_{microscope}"
            self.distance_metric = None
        elif microscope is not None:
            self.group_based = True
            self.image_based = True
            self.group_type = "microscope_type"
            self.group_name = microscope
            self.distance_metric = "lpips"
        elif specimen is not None:
            self.group_based = True
            self.image_based = True
            self.group_type = "object"
            self.group_name = specimen
            self.distance_metric = "mean_gradient"
        else:
            self.group_based = False
            self.image_based = True
            self.group_type = None
            self.group_name = None
            self.distance_metric = "umap"

        assert(self.group_name is not None if self.group_based else True)
        assert(self.group_type is not None if self.group_based else True)
        assert(self.distance_metric is not None if self.image_based else True)
