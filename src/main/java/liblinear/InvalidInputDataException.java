package liblinear;


public class InvalidInputDataException extends Exception {

    private static final long serialVersionUID = 2945131732407207308L;

    private final String      _filename;

    private final int         _line;

    public InvalidInputDataException( String message, String filename, int line ) {
        super(message);
        _filename = filename;
        _line = line;
    }

    public InvalidInputDataException( String message, String filename, int lineNr, NumberFormatException cause ) {
        super(message, cause);
        _filename = filename;
        _line = lineNr;
    }

    public String getFilename() {
        return _filename;
    }

    public int getLine() {
        return _line;
    }

    @Override
    public String toString() {
        return super.toString() + " (" + _filename + ":" + _line + ")";
    }

}
