package de.bwaldvogel.liblinear;


public class InvalidInputDataException extends Exception {

    private static final long serialVersionUID = 2945131732407207308L;

    private final int         _line;

    public InvalidInputDataException(String message, int line) {
        super(message);
        _line = line;
    }

    public InvalidInputDataException(String message, int lineNr, Exception cause) {
        super(message, cause);
        _line = lineNr;
    }

    public int getLine() {
        return _line;
    }

    @Override
    public String toString() {
        return super.toString() + " (line " + getLine() + ")";
    }

}
