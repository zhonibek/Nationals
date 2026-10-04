import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

public class CompileAndRun {
    public static void main(String[] args) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("BLOCKED: Java compiler unavailable");
        List<String> options = new ArrayList<>(Arrays.asList(
                "-encoding", "UTF-8", "-proc:none", "-classpath", args[0], "-sourcepath", args[0], "-d", args[0]));
        options.addAll(Arrays.asList(args).subList(1, args.length));
        int result = compiler.run(null, System.err, System.err, options.toArray(new String[0]));
        if (result != 0) throw new IllegalStateException("BLOCKED: Pedro compilation failed: " + result);
        try (URLClassLoader loader = new URLClassLoader(new URL[] {Paths.get(args[0]).toUri().toURL()}, null)) {
            loader.loadClass("PedroProbe").getMethod("main", String[].class).invoke(null, (Object) new String[0]);
        }
    }
}
