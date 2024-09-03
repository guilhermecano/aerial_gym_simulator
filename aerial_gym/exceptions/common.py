class InvalidArchitectureException(Exception):
    @classmethod
    def create(cls, num_propellers: int, num_bases: int) -> "InvalidArchitectureException":
        return cls(f"Invalid architecture for {num_propellers} propellers and {num_bases} bases.")